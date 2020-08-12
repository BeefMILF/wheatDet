import pandas as pd
from pathlib import Path
import numpy as np
from glob import glob
from tqdm import tqdm
import yaml


def split_bbox_column(images: pd.DataFrame):
    """ split bbox column """
    images = images.copy()
    bbox_items = images.bbox.str.split(',', expand=True)
    images['bbox_xmin'] = bbox_items[0].str.strip('[ ').astype(float)
    images['bbox_ymin'] = bbox_items[1].str.strip(' ').astype(float)
    images['bbox_width'] = bbox_items[2].str.strip(' ').astype(float)
    images['bbox_height'] = bbox_items[3].str.strip(' ]').astype(float)
    return images


def main():
    with open(config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    train_fns = list(Path(hparams['train_image_path']).rglob('*.jpg'))

    # train_fns = glob(hparams['train_image_path'] + '/*')
    # test_fns = glob(hparams['test_image_path'] + '/*')

    train = pd.read_csv(hparams['train_annotation_path'])

    # dataframe with all images
    train_images = pd.DataFrame([fns.stem for fns in train_fns])
    train_images.columns = ['image_id']

    train_images = train_images.merge(train, on='image_id', how='left')
    train_images.bbox = train_images.bbox.fillna('[0,0,0,0]')

    train_images = split_bbox_column(train_images)

    print(train_images.head())



if __name__ == '__main__':
    config_path = 'retinaface/configs/2020-07-20.yaml'
    main()