from typing import Dict, Any, List

import torch
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from os import mkdir
import yaml
import pathlib
import tqdm
from sklearn.manifold import TSNE

from models.yolo import Model
import detect
from train_tl import filter_image_labels
from utils.datasets import create_dataloader


class LayerLogger:
    def __init__(self):
        self.logs: List[torch.Tensor] = []

    def log(self, module: torch.nn.Module, module_input, module_output):
        if isinstance(module_output, torch.Tensor):
            self.logs.append(module_output.detach().cpu())


def coll_string_to_npy_array_int(list_str: str):
    return np.fromstring(list_str[1:-1], sep=',', dtype=int)


def coll_string_to_npy_array_float(list_str: str):
    return np.fromstring(list_str[1:-1], sep=',', dtype=float)


def per_class_arrays(data_df: pd.DataFrame, index_col: str, data_col: str):
    assert data_df[data_col].dtype == object
    max_len = max(len(row) for row in data_df[data_col])
    class_lists = [[] for _ in range(max_len)]
    for idx, data in data_df.filter(items=[index_col, data_col]).itertuples(index=False, name=None):
        for cls, val in enumerate(data):
            class_lists[cls].append((idx, val))

    return [np.array(lst) for lst in class_lists]


def add_gen_columns(df: pd.DataFrame):
    df['total_mistakes'] = np.array([np.sum(val) for val in df['mistake_counts']], dtype=int)
    df['total_labels'] = np.array([np.sum(val) for val in df['label_counts']], dtype=int)
    df['mistake_rates'] = [mistakes / counts for mistakes, counts in zip(df['mistake_counts'], df['label_counts'])]
    df['mistake_rate_avg'] = df['total_mistakes'] / df['total_labels']

    df['iou_sum'] = np.array([np.sum(val) for val in df['iou_totals']], dtype=float)
    df['iou_norm'] = [ious / counts for ious, counts in zip(df['iou_totals'], df['label_counts'])]
    df['iou_norm_sum'] = df['iou_sum'] / df['total_labels']


def per_class_line_graph(df: pd.DataFrame, col_name: str, file_path: str, ylimits=(0.0, None)):
    per_class_mistakes_train = per_class_arrays(df, 'epoch', col_name)
    plt.figure()
    for cls, cls_arr in enumerate(per_class_mistakes_train):
        plt.plot(cls_arr[:, 0], cls_arr[:, 1], label=model.names[cls])
    plt.ylim(bottom=ylimits[0], top=ylimits[1])
    plt.legend()
    plt.savefig(file_path)


log_dir = './runs/train_tl/exp11'
out_dir = f'{log_dir}/plots'
model_path = f'{log_dir}/epoch-98.pt'
sample_val_ds = './data/VOC.yaml'
input_img_size = 640
batch_size = 64
device = torch.device('cuda:0')

if __name__ == '__main__':
    try:
        mkdir(out_dir)
    except FileExistsError:
        pass

    common_cvt_dict = {'mistake_counts': coll_string_to_npy_array_int, 'iou_totals':
                       coll_string_to_npy_array_float, 'class_set': coll_string_to_npy_array_int,
                       'label_counts': coll_string_to_npy_array_int}

    train_df: pd.DataFrame = pd.read_csv(log_dir + '/train.csv', converters=common_cvt_dict)
    val_df: pd.DataFrame = pd.read_csv(log_dir + '/val.csv', converters=common_cvt_dict)
    add_gen_columns(train_df)
    add_gen_columns(val_df)

    model_info: Dict[str, Any] = torch.load(model_path)
    model: Model = model_info['model'].to(device)

    plt.figure()
    plt.plot(train_df['epoch'], train_df['total_losses'], label='Total training losses')
    plt.plot(val_df['epoch'], val_df['total_losses'], label='Total validation losses')
    plt.ylim(bottom=0.0)
    plt.legend()
    plt.savefig(out_dir + '/losses.png')

    plt.figure()
    plt.plot(train_df['epoch'], train_df['mistake_rate_avg'], label='Training classification errors per true label')
    plt.plot(val_df['epoch'], val_df['mistake_rate_avg'], label='Validation classification errors per true label')
    plt.ylim((0.0, 2.0))
    plt.legend()
    plt.savefig(out_dir + '/mistakes.png')

    per_class_line_graph(train_df, 'mistake_rates', out_dir + '/per_class_mistakes_train.png', ylimits=(0.0, 2.0))
    per_class_line_graph(val_df, 'mistake_rates', out_dir + '/per_class_mistakes_val.png', ylimits=(0.0, 2.0))

    plt.figure()
    plt.plot(train_df['epoch'], train_df['iou_norm_sum'], label='Average IOU per true label in training')
    plt.plot(val_df['epoch'], val_df['iou_norm_sum'], label='Average IOU per true label in validation')
    plt.legend()
    plt.savefig(out_dir + '/avg_iou.png')

    per_class_line_graph(train_df, 'iou_norm', out_dir + '/per_class_iou_train.png', ylimits=(0.0, 0.7))
    per_class_line_graph(val_df, 'iou_norm', out_dir + '/per_class_iou_val.png', ylimits=(0.0, 0.7))

    plt.figure()
    plt.bar(model.names, train_df['label_counts'].iloc[-1], label='Total number of image labels')
    plt.legend()
    plt.savefig(out_dir + '/per_class_bar.png')

    with open(sample_val_ds, 'r', encoding='utf8') as val_manifest_file:
        val_manifest = yaml.safe_load(val_manifest_file)

    ds_labels_to_model_labels = {next(idx for idx, val in enumerate(val_manifest['names']) if val == this_name): model_idx
                                 for model_idx, this_name in enumerate(model.names)}

    root_ds_path = val_manifest['path']
    val_paths = root_ds_path / pathlib.Path(val_manifest['val']) if isinstance(val_manifest['val'], str) \
                        else [root_ds_path / pathlib.Path(subpath) for subpath in val_manifest['val']]
    # loggers = [LayerLogger() for _ in model.model]
    #
    # for module, logger in zip(model.model, loggers):
    #     module.register_forward_hook(logger.log)

    logger = LayerLogger()
    model.model[-2].register_forward_hook(logger.log)

    val_loader, val_dataset = create_dataloader(val_paths, input_img_size, batch_size, 32, workers=0)

    batch_labels = []

    with torch.no_grad():
        for imgs, targets, _, _ in tqdm.tqdm(val_loader, desc='Calculating layer outputs...'):
            imgs = (imgs.to(device) / 255.0).half()
            translated_targets = filter_image_labels(ds_labels_to_model_labels.keys(), targets)
            translated_targets[:, 1].apply_(ds_labels_to_model_labels.get)

            model.forward(imgs)
            batch_labels.append(translated_targets[:, :2])

    log_tensor = torch.cat(logger.logs, dim=0)
    log_points = torch.reshape(log_tensor, (log_tensor.shape[0], -1)).numpy()

    del logger
    del log_tensor

    img_labels = np.array([torch.mode(batch[batch[:, 0] == i, 1]).values.item() if torch.any(batch[:, 0] == i) else -1
                           for batch in batch_labels for i in range(batch_size)], dtype=int)[:log_points.shape[0]]

    transformed_points = TSNE().fit_transform(log_points)

    plt.figure()
    for category in np.unique(img_labels):
        masked_points = transformed_points[img_labels == category]
        plt.scatter(masked_points[:, 0], masked_points[:, 1], s=2.5, label=model.names[category] if category != -1 else 'No labels')

    plt.legend(loc=(1.04, 0))
    plt.savefig(out_dir + '/tsne_val.png', dpi=300.0, bbox_inches='tight')

    # detect.run(weights=model_path, project=log_dir, source='../datasets/coco/validation/data', name='sample_images')

