# -*- coding: utf-8 -*-

'''
This script can be applied to build a deep neural network with the hidden layers have 
256-128-64-32 units respectively, the activation function for all fully connected layers
are RELU functions. The output of the network has 10 different units and classified
by softmax activation function. The input size can be customized by users.

This network uses the dropout method to prevent the overfitting
The configuration and size of the dropout layers have been adjusted by hands in 
advance to make it more proper for the GTZAN dataset
'''

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Dropout

def buildnet(input_shape,output_shape):
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(output_shape, activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model
