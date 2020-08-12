import argparse
import os
from pathlib import Path
from glob import glob
import yaml

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import cv2


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


# HELPERS FOR DATA PREPROCESSING


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


def add_brightness(df):
    brightness = []
    green = []
    yellow = []
    for _, row in df.iterrows():
        img_id = row.image_id
        image = cv2.imread(os.path.join(hparams['train_image_path'], img_id + '.jpg'))
        brightness.append(get_image_brightness(image))
        # green.append(get_percentage_of_green_pixels(image))
        # yellow.append(get_percentage_of_yellow_pixels(image))

    brightness_df = pd.DataFrame(brightness)
    brightness_df.columns = ['brightness']
    df = pd.concat([df, brightness_df], ignore_index=True, axis=1)
    df.columns = ['image_id', 'brightness']
    return df


def stratified_split_by_col(df, col, bin=100):
    # digitize brightness for stratification
    bins = np.linspace(df[col].min(), df[col].max(), bin)
    y_binned = np.digitize(df[col], bins)
    train_df, test_df = train_test_split(df, test_size=0.03, random_state=hparams['seed'], stratify=y_binned)
    return train_df, test_df


if __name__ == '__main__':
    config_path = 'retinaface/configs/2020-07-20.yaml'
    with open(config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    train_fns = glob(hparams['train_image_path'] + '/*')
    test_fns = glob(hparams['test_image_path'] + '/*')

    train = pd.read_csv(hparams['train_annotation_path'])

    # dataframe with all images
    train_images = pd.DataFrame([Path(fns).stem for fns in train_fns])
    train_images.columns = ['image_id']

    train_images = train_images.merge(train, on='image_id', how='left')
    train_images.bbox = train_images.bbox.fillna('[0,0,0,0]')

    train_images = split_bbox_column(train_images)

    train_images['bbox_area'] = train_images['bbox_width'] * train_images['bbox_height'] / (1024 * 1024)

    # add brightness to the dataframe
    images_df = pd.DataFrame(train_images.image_id.unique())
    images_df.columns = ['image_id']
    brightness_df = add_brightness(images_df)
    # digitize brightness for stratification
    train_df1, val_df1 = stratified_split_by_col(brightness_df, 'brightness')


    train_images = train_images.merge(brightness_df, on='image_id')

    # nobbox_inds = train_images.bbox_area == 0
    # bigbbox_inds = train_images.bbox_area > 0.1
