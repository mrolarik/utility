# example of loading an image with the Keras API
from time import time
from pathlib import Path
import numpy as np
import gc

import os
import pickle
from collections import defaultdict

from PIL import Image
import matplotlib.pyplot as plt
from sklearn import preprocessing
from skimage import transform
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.utils import to_categorical

from dataclasses import dataclass

@dataclass
class plantData:
  data: int = 0
  data_cnn: int = 0
  data_path: str = 0
  target: int = 0
  new_target: int = 0
  new_target_cnn: int = 0
  target_names: int = 0
  num_classes: int = 0

def plantvillage_to_cnn(img_path, img_size=224, verbose=True):
  plant_data = plantData()
  data = []
  data_path = []
  target = []  

  posix_img_path = list(Path(img_path).rglob('Corn_*/*.[JjpP]*[gG]'))
  #print('Number of images', len(posix_img_path))

  for i in range(len(posix_img_path)):
    target.append(posix_img_path[i].as_posix().split('/')[-2])
    data_path.append(posix_img_path[i].as_posix())

    im = load_img(posix_img_path[i].as_posix())
    im2arr = np.array(im) # im2arr.shape: height x width x channel
    #arr2im = Image.fromarray(im2arr)
    im = None
    #data.append(im2arr)
    #data[i] = transform.resize(im2arr, (img_size, img_size))
    tmp = transform.resize(im2arr, (img_size, img_size))
    data.append(tmp)
    if i % 1000 == 0:
      print('image', i)
      gc.collect()    

  target_names = np.unique(target)
  num_classes = len(np.unique(target))

  #convert target string to int
  le = preprocessing.LabelEncoder()
  new_target = le.fit(target).transform(target)

  # preparing CNN label
  new_target_cnn = to_categorical(new_target, num_classes)

  # preparing CNN data
  data_cnn = np.array(data)
  gc.collect()
  data_cnn = data_cnn.astype('float32')
  data_cnn /= 255.0  

  plant_data.data_cnn = data_cnn
  plant_data.data_path = data_path
  plant_data.target = target 
  plant_data.new_target = new_target
  plant_data.new_target_cnn = new_target_cnn
  plant_data.target_names = target_names 
  plant_data.num_classes = num_classes

  if(verbose):
    print('num_classes:', num_classes)
    print('Number of images:', len(target))
    print('shape of image:', data[0].shape)
    print('target name', np.unique(target_names), sep='\n')
    print('new target name:', np.unique(new_target))

  data = None
  data_cnn = None
  data_path = None
  target = None
  new_target = None
  new_target_cnn = None
  target_names = None
  num_classes = None
  gc.collect()

  return plant_data
