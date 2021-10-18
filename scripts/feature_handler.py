# -*- coding: utf-8 -*-
'''
This script can be applied to extract features from audio files, the extracted features include:
chromatogram CQT, STFT, CENS, Root Mean Square Energy, spectral centroid, spectral bandwidth,
spectral rolloff, zero crossing rate, mel-frequency cepstral coefficients.

This script can extract the feature from single song or from a batch of songs. As our purpose in
this project is to classify the GTZAN songs, if users input the path of where the GTZAN dataset is 
stored into the script, it can extract all song's features and store it as a csv file.

This script uses the librosa python library to extract the features.
A part of the application methods reference: 
https://gist.github.com/sdoshi579/dbabc940cd8af6a1d9e37d2ffe2cb655#file-music-classification-into-genres-py
'''

import librosa
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import shutil

class batch_feature():
    '''
    This class can be applied to extract features from GTZAN dataset, the application method
    is simple by inputing the path of the GTZAN dataset. In addition to the GTZAN dataset,
    other batchs of songs' features can also be extracted by this
    '''
    def __init__(self,path):
        self.path = path
        
    def batch_get_labels(self):
        labels = os.listdir(self.path)
        return labels
    
    # Apply 'feature_extract' function to extract the features from songs in a batch, and store them in a dictionary
    def batch_features_extract(self):
        self.data = {'filename':[], 'chroma_stft':[],'chroma_cqt':[],'chroma_cens':[], 'rmse':[], 'spectral_centroid':[],\
                'spectral_bandwidth':[], 'rolloff':[], 'zero_crossing_rate':[],\
                'mfcc1':[],'mfcc2':[],'mfcc3':[],'mfcc4':[],'mfcc5':[],'mfcc6':[],\
                'mfcc7':[],'mfcc8':[],'mfcc9':[],'mfcc10':[],'mfcc11':[],'mfcc12':[],\
                'mfcc13':[],'mfcc14':[],'mfcc15':[],'mfcc16':[],'mfcc17':[],\
                'mfcc18':[],'mfcc19':[],'mfcc20':[],'label':[]}
        labels = self.batch_get_labels()
        for label in labels:
            wavfiles = [self.path + '/' + label + '/' + wavfile for wavfile in os.listdir(self.path + '/' + label)]
            for wavfile in tqdm(wavfiles, "Saving features of label '{}'".format(label)):
                features_data = feature_extract(wavfile)
                filename = os.path.basename(wavfile)
                self.data['filename'].append(filename)
                for key in features_data:
                    self.data[key].append(features_data[key])
                self.data['label'].append(label)       
        return self.data

    # If users use this function, the above extracted features will be stored in 
    # the input path. If users don't input the path, a csv file will be created in
    # the currently working path
    def batch_save_csv(self, path = None):
        if path is not None:
            dir_path = path
        dir_path = os.getcwd()
        print(dir_path)
        dir_path = f"{dir_path}/GTZAN_features"
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)
        pd.DataFrame.from_dict(data=self.data).to_csv(f"{dir_path}/GTZAN_features.csv", index=False, header=True)


#The following function can extract features from a single song and store them in a dictionary                                    
def feature_extract(Path,Duration=30):
    y, sr = librosa.load(Path, mono=True, duration=Duration)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr) 
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    rmse = librosa.feature.rms(y)        
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)        
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)        
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)        
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)        
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    
    data = {}   
    data['chroma_stft'] = np.mean(chroma_stft)
    data['chroma_cqt'] = np.mean(chroma_cqt)
    data['chroma_cens'] = np.mean(chroma_cens)
    data['rmse'] = np.mean(rmse)
    data['spectral_centroid'] = np.mean(spectral_centroid)
    data['spectral_bandwidth'] = np.mean(spectral_bandwidth)
    data['rolloff'] = np.mean(rolloff)
    data['zero_crossing_rate'] = np.mean(zero_crossing_rate)     
    
    i = 1
    for mfcc in mfccs:
        data['mfcc{}'.format(i)] = np.mean(mfcc)
        i += 1
      
    return data