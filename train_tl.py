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
from utils.loggers.csv import CSVLogger
import val
from utils.metrics import box_iou


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


def most_common_categories(dataset: LoadImagesAndLabels):
    all_labels = np.concatenate([dataset.labels[idx][:, 0] for idx in dataset.indices], axis=0).astype('int64')
    label_counts = np.bincount(all_labels)
    return np.argsort(label_counts)[::-1]


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
            if len(img_label_set) > 0 and not img_label_set.isdisjoint(self.selected_categories):
                self.img_idx_list.append(idx)

        self.num_labels = sum(len(dataset.labels[idx]) for idx in self.img_idx_list)

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.img_idx_list)

    def __len__(self):
        return len(self.img_idx_list)


def filter_image_labels(categories: Iterable[int], label_batch: torch.Tensor):
    labels_in_set = sum(label_batch[:, 1] == item for item in categories).bool()
    return label_batch[labels_in_set]


def resample(items: Sized, count: int):
    if len(items) > count:
        prune_items = set(rng.choice(len(items), len(items) - count, replace=False))
        new_items = [x for i, x in enumerate(items) if i not in prune_items]
        assert len(new_items) == count
        return new_items
    else:
        return list(items)


def values_of(t: torch.Tensor):
    if torch.numel(t) > 1:
        return list(values_of(sub_t) for sub_t in t)
    else:
        return t.item()


def run_forward_pass(model: Model, imgs: torch.Tensor, targets: torch.Tensor, loss_fn: ComputeLoss):
    total_mistakes = torch.zeros((len(used_categories),), dtype=torch.int64, device=imgs.device)
    total_iou_matches = torch.zeros((len(used_categories),), dtype=torch.float, device=imgs.device)

    translated_targets = torch.clone(targets)
    translated_targets[:, 1].apply_(dataset_labels_to_new_labels.get)
    translated_targets = translated_targets.to(device)

    target_xyxys = torch.clone(translated_targets[:, 2:6]) * input_img_size
    target_xyxys[:, 2:4] += target_xyxys[:, 0:2]

    pred, layer_vals = model(imgs)
    loss, loss_items = loss_fn(layer_vals, translated_targets)  # loss scaled by batch_size

    with torch.no_grad():
        processed_pred = non_max_suppression(torch.clone(pred))

        for i, det_list in enumerate(processed_pred):
            true_class_entries = translated_targets[translated_targets[:, 0] == i][:, 1].type(torch.int64)
            pred_class_entries = det_list[:, -1].type(torch.int64)
            true_class_counts = torch.bincount(true_class_entries, minlength=len(used_categories))
            class_counts = torch.bincount(pred_class_entries, minlength=len(used_categories))
            total_mistakes += torch.abs(true_class_counts - class_counts)

            for cls in range(len(used_categories)):
                true_class_filter = true_class_entries == cls
                pred_class_filter = pred_class_entries == cls

                if torch.sum(true_class_filter).item() > 0 and torch.sum(pred_class_filter).item() > 0:
                    iou_mat = box_iou(target_xyxys[translated_targets[:, 0] == i][true_class_filter],
                                      det_list[:, :4][pred_class_filter])

                    max_matches = torch.argmax(iou_mat, dim=1)
                    for this_pred in torch.unique(max_matches):
                        total_iou_matches[cls] += torch.max(iou_mat[max_matches == this_pred, this_pred])

            # print(f'Image {i}: Found {det_list[:, -1]}')

    all_true_class_counts = torch.bincount(translated_targets[:, 1].type(torch.int64), minlength=len(used_categories))
    return loss, total_mistakes, total_iou_matches, all_true_class_counts


device = torch.device('cuda:0')

train_data_file = 'K:\\cse583\\project_data\\VOC\\images\\train2007'
val_data_file = 'K:\\cse583\\project_data\\VOC\\images\\val2007'
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
total_epochs = 80
val_interval = 1
val_size = 10
max_replay_cache_size = 50
online_add_size = 10
rng = np.random.default_rng()

category_add_schedule = lambda epoch: 1 if (epoch + 1) % 20 == 0 else 0

ch = 3
# s = 256

if __name__ == '__main__':
    # torch.set_printoptions(profile='full')
    train_csv_logger = CSVLogger('./train.csv')
    val_csv_logger = CSVLogger('./val.csv')

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
                                                    sampler=lambda dataset: ImageLabelSampler(dataset, set()))
    val_loader, val_dataset = create_dataloader(val_data_file, input_img_size, batch_size, 32, workers=0,
                                                    sampler=lambda dataset: ImageLabelSampler(dataset, set()))
    sorted_categories = most_common_categories(train_dataset)
    used_categories = set(sorted_categories[:offline_categories])

    train_loader.sampler.update_categories(train_dataset, used_categories)
    train_loader.sampler.img_idx_list = resample(train_loader.sampler.img_idx_list, max_replay_cache_size * batch_size)
    used_train_images = set(train_loader.sampler.img_idx_list)

    val_loader.sampler.update_categories(val_dataset, used_categories)
    val_loader.sampler.img_idx_list = resample(val_loader.sampler.img_idx_list, val_size * batch_size)

    dataset_labels_to_new_labels = {ds_label: idx for idx, ds_label in enumerate(used_categories)}

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

    replay_cache = [(imgs, targets, paths) for (imgs, targets, paths, _)
                    in tqdm.tqdm(train_loader, desc='Caching offline training batches...', file=sys.stdout)]


    for epoch in range(total_epochs):
        categories_to_add = category_add_schedule(epoch)

        if categories_to_add > 0:
            print(f'Adding {categories_to_add} {"category" if categories_to_add == 1 else "categories"}...')

            new_category_set = set(sorted_categories[len(used_categories):len(used_categories) + categories_to_add])
            used_categories.update(new_category_set)
            train_loader.sampler.update_categories(train_dataset, used_categories)
            train_loader.sampler.img_idx_list = resample([idx for idx in train_loader.sampler.img_idx_list
                                                          if idx not in used_train_images], online_add_size * batch_size)
            used_train_images.update(train_loader.sampler.img_idx_list)

            val_loader.sampler.update_categories(val_dataset, used_categories)
            val_loader.sampler.img_idx_list = resample(val_loader.sampler.img_idx_list, val_size * batch_size)

            for category in new_category_set:
                assert category not in dataset_labels_to_new_labels
                dataset_labels_to_new_labels[category] = len(dataset_labels_to_new_labels)

            assert len(dataset_labels_to_new_labels) == len(used_categories)

            base_model.model[-1].set_num_classes(len(used_categories))
            loss_fn = ComputeLoss(base_model)
            optimizer = SGD(base_model.parameters(), lr=base_model.hyp['lr0'])
            replay_cache += [(imgs, targets, paths) for (imgs, targets, paths, _)
                             in tqdm.tqdm(train_loader, desc='Reading online training batches...', file=sys.stdout)]

            replay_cache = resample(replay_cache, max_replay_cache_size)

        imgs: torch.Tensor  # Batch of image tensors (batch size * channels * imgsz * imgsz)
        targets: torch.Tensor  # Batch of labels (batch size * 6)

        losses = []
        total_mistakes = torch.zeros((len(used_categories),), dtype=torch.int64, device=device)
        total_labels = torch.clone(total_mistakes)
        total_iou_matches = torch.zeros((len(used_categories),), dtype=torch.float, device=device)

        # train_conf_mat = torch.zeros(offline_categories, offline_categories, dtype=torch.int64)

        for batch_num, (imgs, targets, paths) in enumerate(tqdm.tqdm(replay_cache, desc=f'Training epoch {epoch}...', file=sys.stdout)):  # batch -----------------------------------------------
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
            targets = filter_image_labels(used_categories, targets)
            optimizer.zero_grad()

            loss, mistakes, iou_matches, these_class_counts = run_forward_pass(base_model, imgs, targets, loss_fn)
            total_mistakes += mistakes
            total_iou_matches += iou_matches
            total_labels += these_class_counts
            losses.append(loss)

            loss.backward()
            optimizer.step()

        # total_labels = sum(targets.shape[0] for _, targets, _ in replay_cache)

        log_items = {'epoch': epoch, 'mistake_counts': values_of(total_mistakes),
                     'iou_totals': values_of(total_iou_matches),
                     'label_counts': values_of(total_labels), 'total_losses': sum(losses).item(),
                     'class_set': used_categories}
        train_csv_logger.log(log_items)

        print(f'Epoch {epoch} training: Mistake breakdown: {torch.div(total_mistakes, total_labels)} ({torch.sum(total_mistakes).item()} '
              f'total mistakes, IOU match breakdown: {torch.div(total_iou_matches, total_labels)}, {torch.sum(total_labels).item()} total '
              f'labels, total losses: {sum(losses).item()})')

        if (epoch + 1) % val_interval == 0:
            total_mistakes = torch.zeros((len(used_categories),), dtype=torch.int64, device=device)
            total_iou_matches = torch.zeros((len(used_categories),), dtype=torch.float, device=device)
            total_labels = torch.clone(total_mistakes)
            losses = []

            for batch_num, (imgs, targets, paths, _) in enumerate(tqdm.tqdm(val_loader, desc='Validating model...', file=sys.stdout)):
                imgs = imgs.to(device, non_blocking=True).float() / 255
                targets = filter_image_labels(used_categories, targets)

                with torch.no_grad():
                    loss, mistakes, iou_matches, these_class_counts = run_forward_pass(base_model, imgs, targets, loss_fn)

                total_mistakes += mistakes
                total_iou_matches += iou_matches
                total_labels += these_class_counts
                losses.append(loss)

            print(f'Epoch {epoch} validation: Mistake breakdown: {torch.div(total_mistakes, total_labels)} ({torch.sum(total_mistakes).item()} '
                  f'total mistakes, IOU match breakdown: {torch.div(total_iou_matches, total_labels)}, {torch.sum(total_labels).item()} total '
                  f'labels, total losses: {sum(losses).item()})')

            log_items = {'epoch': epoch, 'mistake_counts': values_of(total_mistakes),
                         'iou_totals': values_of(total_iou_matches),
                         'label_counts': values_of(total_labels), 'total_losses': sum(losses).item(),
                         'class_set': used_categories}
            val_csv_logger.log(log_items)

    # tl_train_cache = [base_model(batch_info[0].to(device).float() / 255).cpu() for batch_info in
    #                   tqdm.tqdm(train_loader, desc="Caching forward values for transfer learning...")]

    train_csv_logger.close()
    val_csv_logger.close()
