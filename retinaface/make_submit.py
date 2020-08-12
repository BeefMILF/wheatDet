import pandas as pd
from pathlib import Path
import json


def make(folder: str):
    file_paths = sorted([x for x in Path(folder).rglob("*") if x.is_file()])
    sbm = pd.read_csv('/home/beefmilf/PycharmProjects/wheatDet/data/sample_submission.csv')

    for f in file_paths:
        with open(f) as out:
            data = json.load(out)
            f_id = Path(data['file_name']).stem
            predstr = ''
            for annot in data['annotations']:

                predstr += ' ' + str(round(annot['score'], 4)) + ' ' + ' '.join([str(b) for b in annot['bbox']])

            sbm.loc[(sbm.image_id == f_id, 'PredictionString')] = predstr.strip()
    sbm.sort_values(by='image_id', inplace=True)
    sbm.to_csv('/home/beefmilf/PycharmProjects/wheatDet/submission.csv', index=False)
    print(sbm)


if __name__ == '__main__':
    test_output = '/home/beefmilf/PycharmProjects/wheatDet/test_output/labels'
    make(test_output)
