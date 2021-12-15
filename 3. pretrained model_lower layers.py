## In this script also I have used a pre-trained VGG16 model, but here I have freezed initial few lower layers(8 layers)
## and trained the remaining layers.

import numpy as np
import pandas as pd


import pickle
import cv2
from skimage import color
import copy
import time
import os

import keras
from keras.models import Model 
from keras.models import Sequential
from scipy.misc import imread
get_ipython().magic('matplotlib inline')
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.layers import Input, Dense, Activation,Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape
from keras.optimizers import SGD
from sklearn.metrics import log_loss

os.chdir('F:\\Img_Class')

## from the previous script (pretrained model), I have already done the cleaning and pre-processing parts
## like retaining only the valid images with valid dimensions in both train and test, normalization, and other steps
## the variables are stored in a pickle file, we load the variables directly over here and proceed for model-building

# Getting back the objects:
with open('pre-trained-model.pkl', 'rb') as f: 
    x_train, y_train, x_test, y_test, _, _ = pickle.load(f)

## hyperparameters of the model:
img_rows, img_cols = 160, 160 # Resolution of the images
channel = 3
num_classes = 5 
batch_size = 32 
nb_epoch = 20


## I had resized my images to 160 X 160, and the VGG16 is built on input image dimensions of 224 X 224
## hence include_top = True would not have accepted my images to the input tensor.
model = VGG16(input_shape = (160,160,3), weights = 'imagenet', include_top = False)
#model.summary()

## it is observed that after each run, the names of the model input and output layers are changed,
## so creating a dictionary to map layer names to the respective layers, rather than hardcoding the label names
layer_dict = dict([(layer.name, layer) for layer in model.layers])

## The VGG16 has a very dense architecture, so I  decide to just include first few layers, and keep them freezed from weight updation.
## Taking the layers till block2_pool only
x = layer_dict['block2_pool'].output

## Defining the customized convolutional network on top of it
x = Convolution2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(5, activation='softmax')(x)

# Creating a new (non-sequential) model.
custom_model = Model(input=model.input, output=x)

# Freezing the pre-trained bottom layers, so they are not trainable(all the layers till the block2_pool)
for layer in custom_model.layers[:7]:
    layer.trainable = False
    
#print(custom_model.layers[6].trainable)

# Learning rate is changed to 0.001
#sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
custom_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Running the model
t = time.time()
custom_model.fit(x_train, y_train,batch_size=batch_size,epochs=nb_epoch,shuffle=True,verbose=1,validation_data=(x_test, y_test))
print('Training time: %s' %(t-time.time()))
(loss, accuracy) = model.evaluate(x_test, y_test, batch_size = 32, verbose = 1)

print('loss=(:.4f), accuracy: {:.4f}%'.format(loss,accuracy * 100))