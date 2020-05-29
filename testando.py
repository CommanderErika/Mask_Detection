# -*- coding: utf-8 -*-
"""
Created on Sun May 24 03:33:26 2020

@author: erika
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import keras
import h5py
from loadDataMask import loadDataMask
from keras.models import Sequential  # criação basica do modelo #
from keras.layers import Dense, Dropout, Activation # Dense é uma das layers para apromorar o aprendizado #
from keras.optimizers import adam # Otimizador, para dizer a taxa de aprendiado do bixo #
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
""" Esse aqui precisamos colocar por que estamos lidando com multi classes"""
from keras.utils.np_utils import to_categorical # quando se está lidnado com mais de uma ou duas classes #
import random

size = 32

x_train, y_train, x_test, y_test = loadDataMask(size) # Carregando os dados #

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)

y_train = to_categorical(y_train, 2) # Categorizando os valores
y_test = to_categorical(y_test, 2) # Categorizando os valores
x_train = x_train/255 # normalizing the values #
x_test = x_test/255 # Normalizando os valores #

# Aqui vai ser o processo de reorganizar os dados para que coloquemos dentro do modelo em formado de Array #

num_pixels = size*size
# x_train = x_train.reshape(x_train.shape[0], num_pixels)
# x_test = x_test.reshape(x_test.shape[0], num_pixels)
x_train = x_train.reshape(x_train.shape[0], size, size, 3)
x_test = x_test.reshape(x_test.shape[0], size, size, 3)
print(x_train.shape)

# Criação da função do nosso modelo #

def createModelMask():
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), input_shape=(size, size, 3), activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), activation = 'relu', padding ='same'))
    model.add(Conv2D(64, (3, 3)))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten()) # é para trasnformar em um Array #
    model.add(Dense(300, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'softmax'))
    model.compile(adam(lr=0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

# criando o modelo #
    
model = createModelMask()

print(model.summary())

model.fit(x_train, y_train, validation_split=0.1, epochs = 30, batch_size = 32, verbose = 1, shuffle = 1)

model.save('testando')