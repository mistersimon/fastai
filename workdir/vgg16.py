# # VGG16 Model
# 
# VGG16 is a pretrained model used in imagenet. Using a pretrained model we have to use the same network architecture as the orginal paper/model
# ![image.png](attachment:image.png)

# The model is exposed by a class wrapped around keras functions. 
# 
# 
# 
# 

# In[1]:




from __future__ import division,print_function

import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt

from numpy.random import random, permutation
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.engine import InputLayer
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image


# In[4]:


def preproc(x):
    """ 
        Completes required preprocessing for VGG16 images
        VGG have mean normalised the data and provide numbers
        VGG inputs colour channels as bgr so it needs to be reversed
        TODO: Make this a class method, keeps giving me an error!
    """
    vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((3, 1, 1))
    x -= vgg_mean
    return x[:,::-1] # reverse axis bgr -> rgb

class VGG_16():
    """Wrapper for the VGG16 model on keras. Useful for finetuning

    Notes:
        Folder structure should have 3 folders, train, valid and test

    Arrtibutes

    """
    # Class variables, too lazy to expose these
    _fpath_vgg16_weights = "./data/vgg16/vgg16.h5"    
    _fpath_class_dict = "./data/vgg16/imagenet_class_index.json"
    
    # Constants relating to image
    VGG_MEAN = np.array([123.68, 116.779, 103.939]).reshape((3, 1, 1))
    IMAGE_SIZE = (224, 224)
    INPUT_SHAPE = (3, 224, 224)
        
    def __init__(self, generators, batch_size=4, tune=True):

        # default parameters
        self.lr = 0.001
        self.loss = 'categorical_crossentropy'


        # Get image generators
        self.gen_train = generators['train']
        self.gen_valid = generators['valid']
        self.gen_test = generators['test']
        
        self.create()

        if tune == True:
            self.finetune()
    
    @classmethod
    def get_generator(self, dir, batch_size=4, shuffle=False, class_mode='categorical'):
        """
            Takes a path, and creates a geneator that provides batches indefinietlt.
            Does som augmentation/normalisation
        """
        print(dir)
        return image.ImageDataGenerator().flow_from_directory(dir, 
                                                               batch_size=batch_size,
                                                               shuffle=shuffle,
                                                               class_mode=class_mode,
                                                               target_size=self.IMAGE_SIZE)

    def create(self):
        """ Intialises a keras model that mimics the VGG16 model. It loads the trained weights"""
        self.model = Sequential()
        
        # Define model as per original VGG16 model
        self.model.add(Lambda(preproc, input_shape=self.INPUT_SHAPE ))

        self._ConvLay(2, 64)
        self._ConvLay(2, 128)
        self._ConvLay(3, 256)
        self._ConvLay(3, 512)
        self._ConvLay(3, 512)

        self.model.add(Flatten())

        self._ConnLay(4096, 0.5)
        self._ConnLay(4096, 0.5)                     
        self._ConnLay(1000, activation='softmax')

        # Load trained weights
        self.model.load_weights(self._fpath_vgg16_weights)
        
        # load imagenet classes index file
        with open(self._fpath_class_dict) as f:
            class_dict = json.load(f)
            self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]
        
        # Note, no need to compile to predict
    
    def finetune(self):
        
        classes_nb = self.gen_train.nb_class
        classes_indicies = self.gen_train.class_indices

        # Replace last layer to adapt to problem, lock others
        self.model.pop()
        for layer in self.model.layers:
            layer.trainable = False
        self._ConnLay(classes_nb, activation='softmax')
        
        self.compile()
        
        # Get classes for current problem
        classes = list(iter(self.gen_train.class_indices))
        
        # Sort classes based on orders by batches
        for c in classes_indicies:
            classes[classes_indicies[c]] = c
        
        # Reset classes stored in model
        self.classes = classes

    def compile(self):
        """ See Keras Documentation
        """
        self.model.compile(optimizer=Adam(lr=self.lr),
                           loss=self.loss, metrics=['accuracy'])
     
    def fit_gen(self, nb_epoch=1):
        return self.model.fit_generator(self.gen_train,
                                 nb_epoch = nb_epoch,
                                 samples_per_epoch = self.gen_train.nb_sample,
                                 validation_data = self.gen_valid,
                                 nb_val_samples = self.gen_valid.nb_sample)
       
    def predict_gen(self, generator):
        """Predicts the labels of a image generator"""
        
        
        
        # Get predictions
        preds =  self.model.predict_generator(generator, generator.nb_sample)
        
        # get index of the most likely class of each image
        #inds = np.argmax(preds, axis=1)
        
        # get the highest probabilty for each image
        # preds_max = [preds_all[i, inds[i]] for i in range(len(inds))]
        
        # get the most likely class for each image
        #pred_class = [self.classes[i] for i in inds]
        
        return preds
    
    def predict(self, imgs):
        """Predicts the labels of a batch of images"""
        
        #predict probabily of each class of each image
        preds_all = self.model.predict(imgs)
        
        # get index of the most likely class of each image
        inds = np.argmax(preds_all, axis=1)
        
        # get the highest probabilty for each image
        # preds_max = [preds_all[i, inds[i]] for i in range(len(inds))]
        
        # get the most likely class for each image
        pred_class = [self.classes[i] for i in inds]
        
        return pred_class

    
    def _ConvLay(self, layers, filters):
        """Addes a convulational layer, acts directly on model object

        Args:
            layers: Number of layers to convolutional layer
            filters: number of filters
        """
        for i in range(layers):
            self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Convolution2D(filters, 3, 3, activation='relu'))

        self.model.add(MaxPooling2D( (2, 2), strides=(2, 2) ))
    
    def _ConnLay(self, neurons, dropout=None, activation='relu'):
        """Addes a fully connected layer, acts directly on model object"""
        self.model.add( Dense(neurons, activation=activation) )

        if dropout != None: self.model.add( Dropout(dropout) ) 
    
    @staticmethod
    def _preproc(x):
        x -= self.VGG_MEAN
        return x[:,::-1] # reverse axis bgr -> rgb