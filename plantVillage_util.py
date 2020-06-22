# example of loading an image with the Keras API
%matplotlib inline
from time import time
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.utils import to_categorical


def show_random_image(x_train, y_train, predicted_classes = '', show_output=False):
  idxs = np.random.randint(x_train.shape[0], size=25)
  images = x_train[idxs]

  if(show_output):
    true_labels = np.argmax(y_train[idxs], axis=1)
    preds = np.argmax(predicted_classes[idxs], axis=1)
  else:
    true_labels = y_train[idxs]

  fig, axes = plt.subplots(5,5, figsize=(8,9))
  for i, ax in enumerate(axes.flat):
    ax.imshow(images[i])
    ax.axis('off')
    if(show_output):
      idx = preds[i]
      color = 'b' if idx == true_labels[i] else 'r'
      tmp = 'a: ' + str(true_labels[i]) + ', p: ' + str(idx)
      ax.set_title(tmp, color=color, fontsize=16)
    else:
      idx = np.argmax(true_labels[i])
      #ax.set_title(target_names[idx])
      ax.set_title(idx)

  plt.show()


def setup_load_plantvillage(img_path, img_query, verbose=True):
  data = []
  data_path = []
  target = []

  #posix_img_path = list(Path(img_path).rglob('Corn_*/*.[JjpP]*[gG]'))
  posix_img_path = list(Path(img_path).rglob(img_query + '_*/*.[JjpP]*[gG]'))
  #print('Number of images', len(posix_img_path))

  for i in range(len(posix_img_path)):
    target.append(posix_img_path[i].as_posix().split('/')[-2])
    data_path.append(posix_img_path[i].as_posix())

    im = load_img(posix_img_path[i].as_posix())
    im2arr = np.array(im) # im2arr.shape: height x width x channel
    #arr2im = Image.fromarray(im2arr)
    data.append(im2arr)

  target_names = np.unique(target)
  num_classes = len(np.unique(target))

  if(verbose):
    print('number of image', len(target))
    print('number of class', num_classes)
    print('shape of data', data[0].shape)

  return data, target, target_names, data_path, num_classes


def convert_plantvillage_target(target, verbose=False):
  le = preprocessing.LabelEncoder()
  new_target = le.fit(target).transform(target)

  if(verbose):
    print('new target_names', np.unique(new_target))

  return new_target


def preparing_cnn_data(data, target_names, new_target):
  # preparing CNN label
  num_classes = len(target_names)
  new_target_cat = to_categorical(new_target, num_classes)

  # preparing CNN data
  data_cnn = np.array(data)
  data_cnn = data_cnn.astype('float32')
  data_cnn /= 255.0

  return new_target_cat, data_cnn
