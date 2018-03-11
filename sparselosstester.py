from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import math
import cv2
import glob
import os
from keras.layers import Input, Dense, Dropout, Flatten, Activation, BatchNormalization, Reshape, Multiply
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Lambda, SeparableConv2D, GlobalAveragePooling2D
from keras.layers.noise import GaussianNoise
from keras.models import Model, Sequential, Input
from keras.optimizers import Adam , SGD , rmsprop
from keras.activations import relu
from keras import regularizers
from keras.regularizers import l1
import json
import random
import keras
import keras.backend as K

for name in glob.glob("*.png"):
	os.remove(name)

def focal_loss(target, output, gamma=2):
    output /= K.sum(output, axis=-1, keepdims=True)
    eps = K.epsilon()
    output = K.clip(output, eps, 1. - eps)
    return -K.sum(K.pow(1. - output, gamma) * target * K.log(output),
                  axis=-1)
def minimize(outputs):
    return 100 * K.sum(K.abs(outputs))

def scalesparse(outputs):
	positives=K.abs(outputs)
	sqrted=K.sqrt(positives)
	summed=K.mean(positives, axis=[2, 3])
	summedsqrt=K.mean(sqrted, axis=[2, 3])
	summedsqrtsquared=K.square(summedsqrt)
	loss=summedsqrtsquared/summed + .1*summed
	loss=K.mean(loss)
	return .02*loss

def dropin(layer):
	sameshape=K.cast(K.greater_equal(layer,-1), dtype=K.floatx())
	droppedones=K.dropout(sameshape,level=.9)
	layer=layer+droppedones
	return layer

array=""
with open('/home/elchanan/keraslearning/CLEVR_v1.0/scenes/CLEVR_val_scenes.json', 'r') as f:
    array = json.load(f)

sx=80
sy=120
s=10
contains=[]
doesntcontain=[]
target=[]
hasbuffer=[]
hasntbuffer=[]
buffcount=0
savenum=0
for n in range(0,1000):
	hasnt=True 
	for k in array['scenes'][n]['objects']:
		if k['shape']=='cube' and k['size']=='large':
			hasnt=False
	name="/home/elchanan/keraslearning/CLEVR_v1.0/images/val/"+str(n).zfill(6)+"shrunk.png"
	if hasnt==True:
		doesntcontain.append(misc.imread(name))
	else:
		contains.append(misc.imread(name))
		target.append(np.zeros((sx,sy,1)))
	print(n)


depth=4

savenum=0
slowsgd=SGD(lr=.001)
def createpairs(size,num):
	global sx,sy,np
	hardcircles=np.zeros((num,sx,sy,4),dtype=np.float32)
	circles=np.zeros((num,sx,sy,4),dtype=np.float32)
	rect=np.zeros((num,sx,sy,4),dtype=np.float32)	
	targetindexs=[]
	for i in range(num):
		hasindex=randint(0,len(contains))
		hasname=contains[hasindex]
		targetindexs.append(hasindex)
		hasntname=doesntcontain[randint(0,len(doesntcontain))]
		circles[i]=hasname/255
		rect[i]=hasntname/255
	return circles,rect,targetindexs

def addrect(pic,x,y):
	size=randint(10,25)
	copy=np.copy(pic)
	for i in range(x-size,x+size):
		for j in range(y-size,y+size):
			if(i>sx-2):
				i=sx-1
			if(i<0):
				i=0
			if(j>sy-2):
				j=sy-1
			if(j<0):
				j=0
			copy[i][j][0]=0
			copy[i][j][1]=0
			copy[i][j][2]=0
			copy[i][j][3]=0
	#copy=np.zeros_like(copy)
	return copy

w = [
    [.1,.1,.1],
    [.1,.3,.1],
    [.1,.1,.1]
    ]

"""w = [
    [0,0,0],
    [0,1,0],
    [0,0,0]
    ]"""

discrimin=Input(shape=(sx,sy,4))
d=discrimin
d=Conv2D(32,(3,3),padding='same')(d)
d=BatchNormalization()(d)
d=Activation('relu')(d)
blurnum=0
for i in range(1,depth): 
	j=i
	if(j>4):
		j=4 #change if want to limit
	d=Conv2D(32*(2**j),(3,3),padding='same')(d)
	d=BatchNormalization()(d)
	d=Activation('relu')(d)
	d=Conv2D(32*(2**j),(3,3),padding='same')(d)
	d=BatchNormalization()(d)
	d=Activation('relu')(d)
	d=Conv2D(32*(2**j),(2,2),padding='same',strides=(2,2))(d)
	d=BatchNormalization()(d)
	d=Activation('relu')(d)
	tokeep=Conv2D(1,(1,1),padding='same')(d) #,activity_regularizer=minimize
	tokeep=Activation('sigmoid')(tokeep)
	tokeep=Conv2D(1,(3,3),padding='same', trainable=False, name="blur"+str(blurnum),activity_regularizer=scalesparse)(tokeep)
	blurnum+=1
	#tokeep=Lambda(lambda x: (x -.5)*2)(tokeep)
	tokeep=Dropout(.10)(tokeep)
	tokeep=Lambda(dropin)(tokeep)
	d=Multiply()([d,tokeep])
d=Conv2D(1,(3,3))(d)
ifis=GlobalAveragePooling2D()(d)
d=Activation('sigmoid')(d)
discriminator=Model(inputs=discrimin,outputs=ifis)
discriminator.compile(loss="mse", optimizer='adam')
discriminator.summary()

for i in range(blurnum):
	layer=discriminator.get_layer("blur"+str(i))
	l=layer.get_weights()
	print(l[0].shape)
	for i in range(len(w)):
		for j in range(len(w)):
			l[0][i][j][0][0]=w[i][j]
	layer.set_weights(l)

inp = discriminator.input                                           # input placeholder
outputs = [layer.output for layer in discriminator.layers]          # all layer outputs
functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function
#convx=discriminator.layers[-1].output.shape[1]
#convy=discriminator.layers[-1].output.shape[2]
"""for count in range(10000):
	c=createpairs(s,8)
	loss=generator.train_on_batch(c[0],c[1])
	print(loss)
	if(count%100==0): # show every 50 iterations
		print("")
		show(4)"""
def uncover(covered,pic,x,y,fineness):
	covered=np.copy(covered)
	for i in range(sx*x//fineness,sx*(x+1)//fineness):
		for j in range(sy*y//fineness,sy*(y+1)//fineness):
			covered[i][j]=pic[i][j]
	return covered

def show(n): # show images ===fix
	b=createpairs(s,n)
	targets=np.squeeze(b[1])
	inputs=np.squeeze(b[0])
	outs=np.squeeze(discriminator.predict(b[0]))
	noouts=np.squeeze(discriminator.predict(b[1]))
	layer_outs = functor([inputs, 1.])
	flatouts=[]
	gatinglayers=[]
	for l in layer_outs:
		if(len(l.shape)==4):
			#print("hi")
			if(l.shape[3]==1):
				flatouts.append(l)
	for  i in range(len(flatouts)):
		if(i%5==2):
			gatinglayers.append(np.squeeze(flatouts[i]))
	#print(gatinglayers)
	for i in range(n):
		plt.subplot(4,n,i+1)
		plt.imshow(inputs[i],vmin=0,vmax=1)
		plt.subplot(4,n,i+5)
		plt.imshow(targets[i],vmin=0,vmax=1)
	for i in range(len(gatinglayers)):
		plt.subplot(4,n,i+9)
		plt.imshow(gatinglayers[i][0],vmin=0,vmax=1)
		"""plt.subplot(4,n,i+9)
		plt.imshow(outs[i],vmin=0,vmax=1)
		plt.subplot(4,n,i+13)
		plt.imshow(noouts[i],vmin=0,vmax=1)"""
	#plt.show()
	global savenum
	plt.savefig(str(savenum)+".png")
	savenum+=1

for count in range(500000):
	num=8
	pairs=createpairs(s,num)
	trainhas=pairs[0]
	trainhasnt=pairs[1]
	trainhas=np.array(trainhas)
	trainhasnt=np.array(trainhasnt)
	images=np.concatenate([trainhas,trainhasnt])
	#labels=np.concatenate([np.ones((len(trainhas),convx,convy,1)),np.zeros((len(trainhasnt),convx,convy,1))])
	labels=np.concatenate([np.ones(len(trainhas)),np.zeros(len(trainhasnt))])
	discrimloss=0
	discrimloss=discriminator.train_on_batch(images,labels) # train the discriminator
	print("classifier" + str(discrimloss))
	if(count%100==0):
		print("")
		show(4)
