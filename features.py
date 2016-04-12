import numpy as np
import pandas as pd 
import h5py
import time
from scipy import stats


DATA_PATH = "data/"

## Load train image features
f = h5py.File(DATA_PATH + 'train_image_features.h5','r')
f_img_id = f['img_id']
train_image_ids = np.copy(f_img_id)
f_label = f['label']
train_image_label = np.copy(f_label)
f_feature = f['feature']
features = np.copy(f_feature)
f.close()

df = pd.DataFrame(data = {'img_id':train_image_ids, 'label':train_image_label})
df_f = pd.DataFrame(data = features)
objs = [df, df_f]
df = pd.concat(objs, axis = 1)

with open(DATA_PATH + 'train_features.csv','w') as f:  
    df.to_csv(f, index=False)

## finished train

## Load train test features
f = h5py.File(DATA_PATH + 'test_image_features.h5','r')
f_img_id = f['img_id']
train_image_ids = np.copy(f_img_id)
f_feature = f['feature']
features = np.copy(f_feature)
f.close()

df = pd.DataFrame(data = {'img_id':train_image_ids})
df_f = pd.DataFrame(data = features)
objs = [df, df_f]
df = pd.concat(objs, axis = 1)

with open(DATA_PATH + 'test_features.csv','w') as f:  
    df.to_csv(f, index=False)
