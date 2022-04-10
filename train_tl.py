import itertools
import sys
from typing import List, Iterable, Iterator, Optional, Sized, Set

import numpy as np
import torch
import torch.utils.data
import tqdm
from torch.optim import SGD
from torch.utils.data.sampler import T_co
import yaml

from models.yolo import Model, Detect
from utils.datasets import create_dataloader, LoadImagesAndLabels
from utils.general import check_img_size, non_max_suppression
from utils.loss import ComputeLoss
import val


def round_robin(sources: Iterable[Iterable]):
    active_sources = [iter(src) for src in sources]

    while len(active_sources) > 0:
        completed = []

        for source in active_sources:
            try:
                yield next(source)
            except StopIteration:
                completed.append(source)

        active_sources = [src for src in active_sources if src not in completed]

def take(items: Iterable, n: int, skip_remainder: bool = False):
    it = iter(items)

    while True:
        these_items = list(itertools.islice(it, n))

        if len(these_items) < n:
            if not skip_remainder and len(these_items) > 0:
                yield these_items
            break
        else:
            yield these_items


def item_labeler(collection, label):
    return ((label, item) for item in collection)


def most_common_categories(dataset: LoadImagesAndLabels, num_categories: int):
    all_labels = np.concatenate([dataset.labels[idx][:, 0] for idx in dataset.indices], axis=0).astype('int64')
    label_counts = np.bincount(all_labels)
    return set(np.argsort(label_counts)[-num_categories:])


class ImageLabelSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LoadImagesAndLabels, selected_categories: Set[int]) -> None:
        super().__init__(dataset)
        self.update_categories(dataset, selected_categories)

        # batches = []
        # for indices in take(img_idx_list, batch_size, skip_remainder=True):
        #     # id_vec = np.concatenate([np.repeat(i, dataset.labels[idx].shape[0]) for i, idx in enumerate(indices)])[:,
        #     #          np.newaxis]
        #     # combined_labels = np.concatenate([dataset.labels[idx] for idx in indices], axis=0)
        #     batches.append(indices)  # (indices, np.concatenate([id_vec, combined_labels], axis=1)))

    def update_categories(self, dataset: LoadImagesAndLabels, selected_categories: Set[int]):
        self.selected_categories = selected_categories

        self.img_idx_list = []
        for idx in dataset.indices:
            img_label_set = set(dataset.labels[idx][:, 0])
            if len(img_label_set) > 0 and img_label_set.issubset(self.selected_categories):
                self.img_idx_list.append(idx)

        self.num_labels = sum(len(dataset.labels[idx]) for idx in self.img_idx_list)

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.img_idx_list)

    def __len__(self):
        return len(self.img_idx_list)


def run_forward_pass(model: Model, imgs: torch.Tensor, targets: torch.Tensor, loss_fn: ComputeLoss):
    total_mistakes = 0

    translated_targets = torch.clone(targets)
    translated_targets[:, 1].apply_(dataset_labels_to_new_labels.get)
    translated_targets = translated_targets.to(device)

    pred, layer_vals = model(imgs)
    loss, loss_items = loss_fn(layer_vals, translated_targets)  # loss scaled by batch_size

    with torch.no_grad():
        processed_pred = non_max_suppression(torch.clone(pred))

        for i, det_list in enumerate(processed_pred):
            true_class_entries = translated_targets[translated_targets[:, 0] == i][:, 1].type(torch.int64)
            true_class_counts = torch.bincount(true_class_entries, minlength=offline_categories)
            class_counts = torch.bincount(det_list[:, -1].type(torch.int64), minlength=offline_categories)
            total_mistakes += torch.sum(torch.abs(true_class_counts - class_counts))
            # print(f'Image {i}: Found {det_list[:, -1]}')

    return loss, total_mistakes


device = torch.device('cuda:0')

train_data_file = 'K:\\cse583\\project_data\\coco\\train2017.txt'
val_data_file = '../datasets/coco/val2017.txt'
base_model_path = './models/yolov5s_tl_base.yaml'
base_weights_path = './yolov5s.pt'

detection_anchors = [
    [10,13, 16,30, 33,23],  # P3/8
    [30,61, 62,45, 59,119],  # P4/16
    [116,90, 156,198, 373,326]  # P5/32
]

model_hyp_file = './data/hyps/hyp.scratch-low.yaml'

input_img_size = 640
batch_size = 16
offline_categories = 5
offline_phase_epochs = 9999
val_interval = 1

category_add_schedule = lambda epoch: 1 if (epoch + 1) % 10 == 0 else 0

ch = 3
# s = 256

if __name__ == '__main__':
    saved_model = torch.load(base_weights_path, map_location='cpu')
    saved_model_state = saved_model['model'].float().state_dict()

    base_model = Model(base_model_path, ch=ch)

    with open(model_hyp_file, 'r') as f:
        base_model.hyp = yaml.safe_load(f)

    for v in base_model.modules():
        for p in v.parameters():
            p.requires_grad = False

    module_collection = list(itertools.chain(*(v.modules() for v in base_model.modules() if isinstance(v, Detect))))
    for v in module_collection:
        for p in v.parameters():
            p.requires_grad = True

    load_keys = [key for (key, value) in base_model.named_parameters() if not value.requires_grad]
    load_state = {key: saved_model_state[key] for key in load_keys}
    base_model.load_state_dict(load_state, strict=False)
    base_model.training = True

    base_model.info(verbose=True)
    assert isinstance(base_model.model[-1], Detect)
    base_model.model[-1].training = False
    base_model.model[-1].set_num_classes(offline_categories)
    base_model.info(verbose=True)

    base_model = base_model.to(device)
    # detection_layer = Detect(anchors=detection_anchors, ch=[ch])

    # stride = torch.tensor([s / x.shape[-2] for x in detection_layer.forward(base_model.forward(torch.zeros(1, ch, s, s)))])
    # gs = max(int(stride.max()), 32)  # grid size (max stride)
    # imgsz = check_img_size(input_img_size, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    train_loader, train_dataset = create_dataloader(train_data_file, input_img_size, batch_size, 32, workers=0,
                                                    sampler=lambda dataset: ImageLabelSampler(dataset, most_common_categories(dataset, offline_categories)))
    val_loader, val_dataset = create_dataloader(val_data_file, input_img_size, batch_size, 32, workers=0,
                                                    sampler=lambda dataset: ImageLabelSampler(dataset, train_loader.sampler.selected_categories))

    dataset_labels_to_new_labels = {ds_label: idx for idx, ds_label in enumerate(train_loader.sampler.selected_categories)}

    # all_labels = np.concatenate([dataset.labels[idx][:, 0] for idx in dataset.indices], axis=0).astype('int64')
    # label_counts = np.bincount(all_labels)
    # most_common_categories = set(np.argsort(label_counts)[-offline_categories:])
    #
    # img_idx_list = []
    # for idx in dataset.indices:
    #     img_label_set = set(dataset.labels[idx][:, 0])
    #     if len(img_label_set) > 0 and img_label_set.issubset(most_common_categories):
    #         img_idx_list.append(idx)
    #
    # batches = []
    # for indices in take(img_idx_list, batch_size, skip_remainder=True):
    #     id_vec = np.concatenate([np.repeat(i, dataset.labels[idx].shape[0]) for i, idx in enumerate(indices)])[:, np.newaxis]
    #     combined_labels = np.concatenate([dataset.labels[idx] for idx in indices], axis=0)
    #     batches.append((indices, np.concatenate([id_vec, combined_labels], axis=1)))

    # label_dict = defaultdict(lambda: set())
    # for idx in dataset.indices:
    #     label_list = dataset.labels[idx]
    #     for label in label_list[:, 0]:
    #         label_dict[int(label)].add(idx)
    #
    # sorted_categories = sorted(label_dict.items(), key=lambda x: len(x[1]), reverse=True)[:offline_categories]
    # category_lists = {label: list(items)[:len(sorted_categories[-1][1])] for label, items in sorted_categories}

    loss_fn = ComputeLoss(base_model)
    optimizer = SGD(base_model.parameters(), lr=base_model.hyp['lr0'])

    batch_cache = [(imgs, targets, paths) for (imgs, targets, paths, _)
                   in tqdm.tqdm(train_loader, desc='Caching offline training batches...', file=sys.stdout)]


    for epoch in range(offline_phase_epochs):
        categories_to_add = category_add_schedule(epoch)

        if categories_to_add > 0:
            print(f'Adding {categories_to_add} {"category" if categories_to_add == 1 else "categories"}...')

            offline_categories += categories_to_add
            train_loader.sampler.update_categories(train_dataset, most_common_categories(train_dataset, offline_categories))
            val_loader.sampler.update_categories(val_dataset, train_loader.sampler.selected_categories)

            for category in train_loader.sampler.selected_categories:
                if category not in dataset_labels_to_new_labels:
                    dataset_labels_to_new_labels[category] = len(dataset_labels_to_new_labels)

            assert len(dataset_labels_to_new_labels) == offline_categories

            base_model.model[-1].set_num_classes(offline_categories)
            loss_fn = ComputeLoss(base_model)
            optimizer = SGD(base_model.parameters(), lr=base_model.hyp['lr0'])
            batch_cache = [(imgs, targets, paths) for (imgs, targets, paths, _)
                           in tqdm.tqdm(train_loader, desc='Re-caching offline training batches...', file=sys.stdout)]

        imgs: torch.Tensor  # Batch of image tensors (batch size * channels * imgsz * imgsz)
        targets: torch.Tensor  # Batch of labels (batch size * 6)

        losses = []
        total_mistakes = 0

        # train_conf_mat = torch.zeros(offline_categories, offline_categories, dtype=torch.int64)

        for imgs, targets, paths in tqdm.tqdm(batch_cache, desc=f'Training epoch {epoch}...', file=sys.stdout):  # batch -----------------------------------------------
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
            optimizer.zero_grad()

            loss, mistakes = run_forward_pass(base_model, imgs, targets, loss_fn)
            total_mistakes += mistakes
            losses.append(loss)

            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch} training: Total mistakes: {total_mistakes} ({train_loader.sampler.num_labels} total'
              f'labels, total losses: {sum(losses)}')

        if (epoch + 1) % val_interval == 0:
            total_mistakes = 0
            losses = []
            for imgs, targets, paths, _ in tqdm.tqdm(val_loader, desc='Validating model...', file=sys.stdout):
                imgs = imgs.to(device, non_blocking=True).float() / 255

                with torch.no_grad():
                    loss, mistakes = run_forward_pass(base_model, imgs, targets, loss_fn)

                total_mistakes += mistakes
                losses.append(loss)

            print(f'Epoch {epoch} validation: Total mistakes: {total_mistakes} ({val_loader.sampler.num_labels} total'
                  f'labels, total losses: {sum(losses)}')

    # tl_train_cache = [base_model(batch_info[0].to(device).float() / 255).cpu() for batch_info in
    #                   tqdm.tqdm(train_loader, desc="Caching forward values for transfer learning...")]
