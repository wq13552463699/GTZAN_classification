# -*- coding: utf-8 -*-

'''
This script can be applied to build a fully connected deep neural network with
the hidden layers have 256-128-64-32 units respectively, the activation function
for all fully connected layers are RELU function. The output of the network 
has 10 different units and classified by softmax activation function. The 
input size can be customized by users.

This network does't have any techniques to prevent the overfitting'
'''

from tensorflow.keras import layers
from tensorflow.keras import models

def buildnet(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model