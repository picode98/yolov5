from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from os import mkdir


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
    df['mistake_rate'] = df['total_mistakes'] / df['total_labels']

    df['iou_sum'] = np.array([np.sum(val) for val in df['iou_totals']], dtype=float)
    df['iou_norm'] = np.array([ious / counts for ious, counts in zip(df['iou_totals'], df['label_counts'])], dtype=object)
    df['iou_norm_sum'] = df['iou_sum'] / df['total_labels']


log_dir = '.'
out_dir = './plots'

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

plt.figure()
plt.plot(train_df['epoch'], train_df['total_losses'], label='Total training losses')
plt.plot(val_df['epoch'], val_df['total_losses'], label='Total validation losses')
plt.legend()
plt.savefig(out_dir + '/losses.png')

plt.figure()
plt.plot(train_df['epoch'], train_df['mistake_rate'], label='Training classification errors per true label')
plt.plot(val_df['epoch'], val_df['mistake_rate'], label='Validation classification errors per true label')
plt.legend()
plt.savefig(out_dir + '/mistakes.png')

plt.figure()
plt.plot(train_df['epoch'], train_df['iou_norm_sum'], label='Average IOU per true label in training')
plt.plot(val_df['epoch'], val_df['iou_norm_sum'], label='Average IOU per true label in validation')
plt.legend()
plt.savefig(out_dir + '/avg_iou.png')

per_class_ious_train = per_class_arrays(train_df, 'epoch', 'iou_norm')
plt.figure()
for cls, cls_arr in enumerate(per_class_ious_train):
    plt.plot(cls_arr[:, 0], cls_arr[:, 1], label=f'Class {cls}')
plt.legend()
plt.savefig(out_dir + '/per_class_iou.png')

plt.figure()
plt.bar([str(x) for x in train_df['class_set'].iloc[-1]], train_df['label_counts'].iloc[-1], label='Total number of image labels')
plt.legend()
plt.savefig(out_dir + '/per_class_bar.png')
