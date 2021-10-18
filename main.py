# -*- coding: utf-8 -*-
'''
This is the main script for users. You can use the following code in the command line
to achieve: 1. Generate the features' csv file from the raw audio data; 2. Train the 
classification model, the hyperparameters including epochs,seed,batch-size,validation 
split and test size that can be set by users. 3. Test the trained model. If users input the 
path to the trained model, the test process will based on user's model, otherwise a
previous stored model(in the project's directory) will be applied for testing. 
4. Users can customorize the style of the DNN, you can use either DNN without dropout
or with dropout. 5.If users are not willing to use the exist feature file, the can
load their own file.
'''

from tools import plot
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scripts import feature_handler
from networks import DNN, DNN_with_dropout
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='GTZAN data classicfication')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--generate-features', type=str, default='', help='Use the GTZAN dataset to generate the feature files, the input string should be the \
                    path to the position store the GTZAN feature, the default option is using the feature file in the project. Notice: If you use this\
                        command, the created feature file will cover the previous exist file!')
parser.add_argument('--load-features-path', type=str, default='', help='If the features files is already exist on your local PC, you can use this \
                    command to load your local file. The script will go to the python working path to find the feature file within the project as default option')
parser.add_argument('--epochs', type=int, default=100, metavar='H', help='Iteration epochs')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='The random seed of shuffling the data')
parser.add_argument('--batch-size', type=int, default=64, metavar='B', help='Batch size')
parser.add_argument('--validation_split', type=int, default=0.2, metavar='V', help='Validation split size')
parser.add_argument('--test-size', type=int, default=0.2, metavar='T', help='The size of the test dataset')
parser.add_argument('--disable-dropout', action='store_true', help='If you want to use the DNN without dropout')
parser.add_argument('--test', action='store_true', help='If you choose this option, this script will be only for test')
parser.add_argument('--models', type=str, default='', metavar='M', help='Load existing model, you can input the path to use your onw model. The defult option will\
                        go to the project directory to find the exist model')

#Create a directory to store all training and testing history. The name of the 
#directory is the id
args = parser.parse_args()
results_dir = os.path.join('results', args.id)
os.makedirs(results_dir, exist_ok=True)

#If you want to generate new feature file
if args.generate_features != '' and os.path.exists(args.generate_features):
    batch = feature_handler.batch_feature(args.generate_features)
    batch.batch_features_extract()
    batch.batch_save_csv()
    quit()

#If you have other feature files
if args.load_features_path != '' and os.path.exists(args.load_features_path):
    GTZAN = pd.read_csv(args.load_features_path)
    GTZAN = GTZAN.drop(['filename'],axis=1)
else:
    dir_path = os.getcwd()
    #Load the file
    GTZAN = pd.read_csv(f"{dir_path}/GTZAN_features/GTZAN_features.csv")
    GTZAN = GTZAN.drop(['filename'],axis=1)

#Load the data in the file and the lable
y_org = GTZAN.iloc[:,-1]
X_org = GTZAN.iloc[:,:-1]

#Shuffle the data with the randon seed you input, defult figure is 0
X,y = shuffle(X_org,y_org,random_state=args.seed)

# Preprocessing the data for better convergance and performance
scaler = StandardScaler()
encoder = LabelEncoder()
y = encoder.fit_transform(y)
X = scaler.fit_transform(np.array(X,dtype=float))

#Split the dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size)

#If your purpose is only for testing
if args.test:
    if args.models != '' and os.path.exists(args.models):
        model = load_model(args.models)
    else:
        model_path = os.getcwd()
        model = load_model(f"{model_path}/Trained_model/model_with_dropout.h5")
else:
    if args.models != '' and os.path.exists(args.models):
        model = load_model(args.models)
        
    # Build the model
    else:
        if args.disable_dropout:
            model = DNN.buildnet(X.shape[1])
        else:
            model = DNN_with_dropout.buildnet(X.shape[1], output_shape=10)
        model.summary()
        history = model.fit(X_train,y_train,epochs=args.epochs,batch_size=args.batch_size,validation_split=args.validation_split)
        draw = plot.show_history()
        
        if args.disable_dropout:
            model.save("model.h5")
            #Plot the training curve(Accuracy Curve & Loss Curve)
            draw.show_acc(history.history, path = results_dir)
            draw.show_loss(history.history,path = results_dir)
        else:
            model.save("model_with_dropout.h5")
            draw.show_acc(history.history, path = results_dir,name='Accuracy Curve With Dropout')
            draw.show_loss(history.history,path = results_dir, name='Loss Curve With Dropout')

# Print out the accuracy of testing
results = model.evaluate(X_test, y_test)
print('The tested accuracy is {}'.format(results[1]))