# extract image features and save it to .h5
# Initialize files
import h5py
import os
from os import walk
import caffe 
import numpy as np
import pandas as pd 
DATA_PATH = "data/"
CAFFE_ROOT = "/home/eodonnell/git/caffe/"
  
def extract_features(images, layer = 'fc7'):
    #net = caffe.Net(CAFFE_ROOT + 'models/bvlc_reference_caffenet/deploy.prototxt',CAFFE_ROOT + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',caffe.TEST)
    net = caffe.Net(CAFFE_ROOT + 'models/bvlc_reference_caffenet/deploy.prototxt',CAFFE_ROOT + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel',caffe.TEST)
    
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(CAFFE_ROOT + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB]]

    num_images= len(images)
    net.blobs['data'].reshape(num_images,3,227,227)
    net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), images)
    out = net.forward()

    return net.blobs[layer].data        


f = h5py.File(DATA_PATH + 'train_image_features.h5','w')
f.create_dataset('img_id',(0,), maxshape=(None,),dtype='|S54')
f.create_dataset('label',(0,), maxshape=(None,),dtype='|S54')
f.create_dataset('feature',(0,4096), maxshape = (None,4096))
f.close()

train_folder = DATA_PATH + 'train/'
#train_images = [os.path.join(train_folder, str(x)+'.jpg') for x in train_photos['img_id']]  # get full filename
#train_images = [train_folder + f for f in os.listdir(train_folder)]
image_name_list = []
train_image_list = []
labels = []
dirs = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
for dir in dirs:
    print dir
    for f in os.listdir(train_folder + dir):
        labels.append(dir)
        image_name_list.append(f)
        filename = train_folder + dir + '/' + f
        train_image_list.append(filename)
        
    

num_train = len(train_image_list)
print "Number of training images: ", num_train
batch_size = 50

# Training Images
for i in range(0, num_train, batch_size): 
    image_paths = train_image_list[i: min(i+batch_size, num_train)]
    image_ids = image_name_list[i: min(i+batch_size, num_train)]
    label_ids = labels[i: min(i+batch_size, num_train)]
    features = extract_features(image_paths, layer='fc7')
    num_done = i+features.shape[0]
    print "num done",num_done
    f= h5py.File(DATA_PATH + 'train_image_features.h5','r+')
    f['img_id'].resize((num_done,))
    f['img_id'][i: num_done] = np.array(image_ids)
    
    f['label'].resize((num_done,))
    f['label'][i: num_done] = np.array(label_ids)
    
    f['feature'].resize((num_done,features.shape[1]))
    f['feature'][i: num_done, :] = features
    f.close()
    if num_done%200==0 or num_done==num_train:
        print "Train images processed: ", num_done
