# -*- coding: utf-8 -*-

'''
The following functions can be applied to draw the curves of the training history including
Accuracy Curve and Loss Curve(The name of the curves can be set by users).
If users input a path, 2 pictures will be created in the path, if not, nothing will be 
created, just show the pictures in the python window for reviewing once.
'''

import matplotlib.pyplot as plt

class show_history():
    
    def show_acc(self,Training_history,train='accuracy',validation='val_accuracy',path = None,name='Accuracy Curve'):
        plt.plot(Training_history[train], linestyle='-', color='b')
        plt.plot(Training_history[validation], linestyle='--', color='r')
        plt.title('Accracy curve')
        plt.xlabel('epoch')
        plt.ylabel('train')
        plt.legend(['train', 'validation'], loc = 'lower right')
        
        if path is not None:
            plt.savefig(f"{path}/{name}.jpg")  
        plt.show()
        
    def show_loss(self,Training_history,train='loss',validation='val_loss', path = None, name='Loss Curve'):
        plt.plot(Training_history[train], linestyle='-', color='b')
        plt.plot(Training_history[validation], linestyle='--', color='r')
        plt.title('Loss curve')
        plt.xlabel('epoch')
        plt.ylabel('train')
        plt.legend(['train', 'validation'], loc = 'lower right')
        
        if path is not None:
            plt.savefig(f"{path}/{name}.jpg")         
        plt.show()
        
        