# Global-Wheat-Detection

kaggle - https://www.kaggle.com/c/global-wheat-detection

## Description 

* ssd model 
* random rotate90 and other augmentation for train  
* wandb logger, used pytorch-lightning/pytorch, fp16   

In the making(missed to do): 

* training on 5 folds(clustered by kmeans, previously reduced by PCA)
* tta for test(inference)

## Test results

in `/test_output/viz`

<p float="left">
  <img src="https://github.com/BeefMILF/wheatDet/blob/master/test_output/viz/2fd875eaa.jpg" width="400" height="400">
  <img src="https://github.com/BeefMILF/wheatDet/blob/master/test_output/viz/f5a1f0358.jpg" width="400" height="400">
  <img src="https://github.com/BeefMILF/wheatDet/blob/master/test_output/viz/796707dd7.jpg" width="400" height="400">
  <img src="https://github.com/BeefMILF/wheatDet/blob/master/test_output/viz/51f1be19e.jpg" width="400" height="400">
</p>

## Submission score MAP@0.5-0.7:0.05

**score**: 0.6448

kaggle notebook - https://www.kaggle.com/heorgiibolotov/kernel1a17c2da7e

cut version of [logs](https://github.com/BeefMILF/wheatDet/blob/master/report/report.pdf) while training

## Project structure 

```
├── apex.sh              
├── eda.ipynb   
├── image_plot.py  
├── prepare_data.py  # invalid version 
├── requirements.txt  
├── submission.csv  
├── wandb
├── cp_data_from_hdd.sh  
├── engdata.py # invalid version
├── parse_data.py # invalid version
├── report         

├── retinaface    
  ├── box_utils.py  
  ├── data_augment.py
  ├── inference.py
  ├── make_submit.py
  ├── net.py
  ├── prior_box.py  
  ├── train.py
  ├── configs       
  ├── dataset.py       
  ├── __init__.py 
  ├── multibox_loss.py
  ├── network.py 
  ├── __pycache__   
  ├── utils.py
  
├── test_output
  ├── labels
  ├── viz
  
├── data
  ├── annotations            
  ├── test   
  ├── train.csv 
  ├── train_with_bbox
  ├── sample_submission.csv
  ├── train  
  ├── train.json
  ├── train_without_bbox
```


## Usage

```
# step 1.(in this repo invalid version of the code)
!python engdata.py  

# step 2. 
!python -m retinaface.train -c retinaface/configs/2020-07-20.yaml

# step 3. 
!python -m retinaface.inference -i data/test -c retinaface/configs/2020-07-20.yaml -o test_output -w 2020-07-27/epoch=7.ckpt -v 
```

