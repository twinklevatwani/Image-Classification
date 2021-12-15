## In this script, I have used a pre-trained VGG16 model, where both the train and test images are passed through it.
## The aim of this script is to modify and fine tune the output dense and softmax layer of the pre-trained model. 
## Extracting the features of the train and test images, and use these weights while adding the dense layers to the architecture.

import numpy as np
import pandas as pd
import pickle
import cv2
from skimage import color
import copy
import time
import os

import keras
from keras.models import Sequential
from scipy.misc import imread
get_ipython().magic('matplotlib inline')
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.layers import Dense, Activation

os.chdir('F:\\Img_Class')

# Getting back the objects from the obj.pkl pickle file stored in the local directory:
with open('objs.pkl', 'rb') as f: 
    train_images, train_grey_images, train_label, test_images, test_grey_images, test_label = pickle.load(f)

## there was an error, with converting images to arrays.
## upon inspection, there were certain images which were not in the proper format,ie,
## each image should be an array having 160 rows of 160 (column) values which are a list of 3(RGB channel) values, ie, 160 X 160 X 3.

## The train images which are not of the shape 160 X 160 X 3 are:
#l = []
#for c,single_img in enumerate(train_images):
#    for single_row in single_img:
#        for single_pixel in single_row:
#            if(not isinstance(single_pixel,np.ndarray)):  ## or len(single_pixel) != 3 
#                l.append(c)            
#train_er = list(set(l))

train_valid = []
for c,single_img in enumerate(train_images):
    if(isinstance(single_img[0][0],np.ndarray)):  ## or len(single_pixel) == 3 
                train_valid.append(c)

## similarly, doing the same procedure for the test images as well:
## The test images which are not of the shape 160 X 160 X 3 are:
test_valid = []
for c,single_img in enumerate(test_images):
    if(isinstance(single_img[0][0],np.ndarray)):  ## or len(single_pixel) == 3 
                test_valid.append(c)
                

## As a backup, I had also written all the images into my local directory, and on examining, found out that these were
## those images which were written as - "This photo is no longer available'', and hence probably couldn't get a proper 
## representation of such images in the desired 160 X 160 X 3 configuration.

## So train_er and the test_er contain the indices of those images which have to be discarded from the image arrrays

## building the final list comprehensions, which will retain only the valid images in train and test:

final_train_images = [train_images[i] for i in train_valid]
final_train_label = [train_label[i] for i in train_valid]
final_test_images = [test_images[i] for i in test_valid]
final_test_label = [test_label[i] for i in test_valid]

x_train = final_train_images
y_train = final_train_label
x_test = final_test_images
y_test = final_test_label

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')

# Normalize the color values to 0-1 as they're from 0-255)
x_train /= 255
x_test /= 255


## pretrained model:
# loading VGG16 model weights, pre-trained on the ImageNet dataset.
model = VGG16(weights='imagenet', include_top=False) ## does not include the 3 fully connected layers of the VGG16 confgig

# Extracting features from the train dataset using the VGG16 pre-trained model
features_train=model.predict(x_train)

# Extracting features from the test dataset using the VGG16 pre-trained model
features_test=model.predict(x_test)

with open('pre-trained-model.pkl', 'wb') as f:
    pickle.dump([x_train, y_train, x_test, y_test, features_train, features_test], f)

# Getting back the objects:
#with open('pre-trained-model.pkl', 'rb') as f: 
#    x_train, y_train, x_test, y_test, features_train, features_test = pickle.load(f)


# VERY IMP : flattening the layers to conform to MLP input
train_x=features_train.reshape(features_train.shape[0],12800)
test_x = features_test.reshape(features_test.shape[0],12800)


model=Sequential()

model.add(Dense(800, input_dim=12800, activation='relu',kernel_initializer='uniform'))
keras.layers.core.Dropout(0.3, noise_shape=None, seed=None)

model.add(Dense(200,input_dim=800,activation='sigmoid'))
keras.layers.core.Dropout(0.4, noise_shape=None, seed=None)

model.add(Dense(units=5))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

model.summary()


model.fit(train_x, y_train, epochs=20, batch_size=150, validation_data=(test_x,y_test))


predictions_valid = model.predict(x_test, batch_size=batch_size, verbose=1)

