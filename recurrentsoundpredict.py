from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import math
import cv2
import glob
import os
from keras.layers import Input, Dense, Dropout, Flatten, Activation, BatchNormalization, Reshape, Multiply
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Lambda, SeparableConv2D, GlobalAveragePooling2D, TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.models import Model, Sequential, Input
from keras.optimizers import Adam , SGD , rmsprop
from keras.activations import relu
from keras import regularizers
from keras.regularizers import l1
from keras.layers import LSTM
import json
import random
import keras
import keras.backend as K
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import soundfile as sf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback

filenames=[]
for name in glob.glob("TRAIN/*/*/*"):
	filenames.append(name[:-4])
loadnum=6000
loadnum=min(loadnum,len(filenames)-1)
wavs=[]
maxlen=0

def minimize(outputs):
    return 100 * K.sum(K.abs(outputs))

counter=0
for name in filenames[0:loadnum]:
	counter+=1
	if(counter%100==0):
		print(counter)
	(sig,rate) = sf.read(name+".WAV")
	wavs.append(logfbank(sig,rate))
	if(wavs[-1].shape[0]>maxlen):
		maxlen=wavs[-1].shape[0]

print(maxlen)
filters=wavs[0].shape[1]
allsounds=np.zeros((loadnum,maxlen,filters))

counter=0
for wav in wavs:
	#wav=np.transpose(wav)
	wav=np.copy(wav)
	wav.resize(maxlen,filters)
	allsounds[counter]=wav
	counter+=1
del wavs
mean=np.mean(allsounds)
allsounds-=mean
std=np.std(allsounds)
allsounds/=std
targets=np.copy(allsounds)
targets=np.roll(targets,shift=-1,axis=1)
plt.imshow(targets[0])
plt.show()

HIDDEN_DIM=128
model = Sequential()
#model.add(LSTM(HIDDEN_DIM, input_shape=(None, filters), return_sequences=True))
#for i in range(2):
    #model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(filters), input_shape=(None, filters)))
#model.add(Activation('softmax'))
model.compile(loss="mse", optimizer="rmsprop")

savenum=0
def makeprediction(epoch,logs):
	print("predicting")
	sounds=np.zeros((maxlen,filters))
	sounds[0:5]=allsounds[randint(0,loadnum)][0:5]
	for i in range(4,80):
		print(i)
		prediction=model.predict(np.expand_dims(sounds, axis=0))
		prediction=np.squeeze(prediction)
		sounds[i+1]=prediction[i]
	plt.imshow(sounds)
	global savenum
	plt.savefig(str(savenum)+".png")
	savenum+=1

predict_callback = LambdaCallback(on_epoch_end=makeprediction)
checkpointer = ModelCheckpoint(filepath='linear.hdf5', verbose=1, save_best_only=True)
model.fit(allsounds,targets,
          batch_size=32,
          epochs=70,validation_split=.2,
          callbacks=[checkpointer]) #,predict_callback
