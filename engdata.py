import argparse
import os
from pathlib import Path
from glob import glob
import yaml
import json

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import cv2


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


# boilerplate helpers for feature engineering


def split_bbox_column(images: pd.DataFrame):
    """ split bbox column """
    images = images.copy()
    bbox_items = images.bbox.str.split(',', expand=True)
    images['bbox_xmin'] = bbox_items[0].str.strip('[ ').astype(float)
    images['bbox_ymin'] = bbox_items[1].str.strip(' ').astype(float)
    images['bbox_width'] = bbox_items[2].str.strip(' ').astype(float)
    images['bbox_height'] = bbox_items[3].str.strip(' ]').astype(float)
    return images


def get_image_brightness(image):
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # get average brightness
    return np.array(gray).mean()


def get_percentage_of_green_pixels(image):
    # convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # get the green mask
    hsv_lower = (40, 40, 40)
    hsv_higher = (70, 255, 255)
    green_mask = cv2.inRange(hsv, hsv_lower, hsv_higher)

    return float(np.sum(green_mask)) / 255 / (1024 * 1024)


def get_percentage_of_yellow_pixels(image):
    # convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # get the green mask
    hsv_lower = (25, 40, 40)
    hsv_higher = (35, 255, 255)
    yellow_mask = cv2.inRange(hsv, hsv_lower, hsv_higher)

    return float(np.sum(yellow_mask)) / 255 / (1024 * 1024)


def add_features(df):
    brightness = []
    green = []
    yellow = []
    for _, row in df.iterrows():
        img_id = row.image_id
        image = cv2.imread(os.path.join(hparams['train_image_path'], img_id + '.jpg'))
        brightness.append(get_image_brightness(image))
        green.append(get_percentage_of_green_pixels(image))
        yellow.append(get_percentage_of_yellow_pixels(image))

    features_df = pd.DataFrame([brightness, green, yellow]).T
    features_df.columns = ['brightness', 'green_pixels', 'yellow_pixels']
    df = pd.concat([df, features_df], ignore_index=True, axis=1)
    df.columns = ['image_id', 'brightness', 'green_pixels', 'yellow_pixels']
    return df


def stratified_split_by_col(df, col, bin=20):
    # digitize brightness for stratification
    bins = np.linspace(df[col].min(), df[col].max(), bin)
    y_binned = np.digitize(df[col], bins)

    # Verify the minimum number of groups for any class cannot be less than 2
    values, counts = np.unique(y_binned, return_counts=True)
    np.random.seed(hparams['seed'])
    # train_test_split will raise the error in case bin counts - 1
    for value in values[np.argwhere(counts == 1)]:
        ind = np.random.randint(0, len(y_binned), 10)
        y_binned[ind] = value

    train_df, test_df = train_test_split(df, test_size=0.033, random_state=hparams['seed'], stratify=y_binned)
    return train_df, test_df


# end of boilerplate helpers for feature engineering


def check_bbox(bbox, threshold=1.5):
    x_min, y_min, x_max, y_max = bbox[:4]
    if x_max - x_min < threshold or y_max - y_min < threshold:
        return False
    return True


def toformat(df, image_path):
    data = []
    for img_id, group_df in df.groupby('image_id'):
        image_data = {'file_name': os.path.join(image_path, img_id + '.jpg')}
        annotations = []
        for row_index, row in group_df.iterrows():
            annotation_data = {
                'bbox': [row.bbox_xmin,
                         row.bbox_ymin,
                         row.bbox_xmin + row.bbox_width,
                         row.bbox_ymin + row.bbox_height],
            }
            if check_bbox(annotation_data['bbox']):
                annotations.append(annotation_data)
            else:
                print(f'{annotation_data["bbox"]}, {img_id}')
        image_data['annotations'] = annotations
        data.append(image_data)
    return data


def save_file(data, fname):
    with open(fname, 'w') as f:
        json.dump(data, f)


def prepare_train_val():
    train_fns = glob(hparams['train_image_path'] + '/*')
    train = pd.read_csv(hparams['train_annotation_path_tmp'])

    train_images = pd.DataFrame([Path(fns).stem for fns in train_fns])
    train_images.columns = ['image_id']

    train_images = train_images.merge(train, on='image_id', how='left')
    train_images.bbox = train_images.bbox.fillna('[0,0,0,0]')
    train_images = split_bbox_column(train_images)
    train_images['bbox_ratio'] = train_images['bbox_width'] * train_images['bbox_height'] / (1024 * 1024)

    # remove large boxes
    # (train_images.bbox_ratio > 0)
    train_images = train_images[(train_images.bbox_ratio < 0.1) & (train_images.bbox_ratio > 0)]

    # extract stratified samples for validation
    images_df = pd.DataFrame(train_images.image_id.unique())
    images_df.columns = ['image_id']
    features_df = add_features(images_df)
    train_df1, val_df1 = stratified_split_by_col(features_df[['image_id', 'brightness']], 'brightness')
    train_df2, val_df2 = stratified_split_by_col(features_df[['image_id', 'green_pixels']], 'green_pixels')
    train_df3, val_df3 = stratified_split_by_col(features_df[['image_id', 'yellow_pixels']], 'yellow_pixels')

    val_id_images = pd.concat([val_df1.image_id, val_df2.image_id, val_df3.image_id], ignore_index=True, axis=0).unique()

    # split data into train/val
    inds = train_images.image_id.isin(val_id_images)
    train_images = train_images[['image_id', 'bbox_xmin', 'bbox_ymin', 'bbox_width', 'bbox_height']]
    val_images = train_images[inds]
    train_images = train_images[~inds]

    # save with required format
    val_images = toformat(val_images, hparams['train_image_path'])
    train_images = toformat(train_images, hparams['train_image_path'])

    save_file(val_images, hparams['val_annotation_path'])
    save_file(train_images, hparams['train_annotation_path'])


def prepare_test():
    test_fns = glob(hparams['test_image_path'] + '/*')


if __name__ == '__main__':
    config_path = 'retinaface/configs/2020-07-20.yaml'
    with open(config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    prepare_train_val()
