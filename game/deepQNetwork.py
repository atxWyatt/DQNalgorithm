from keras.models import *
from keras.layers import *
from keras.optimizers import *

from collections import *

import random

import numpy

class deepQNetwork:
	#Paramaters are states, actions, length, gam, eps, activation function, optimization function, training reset size
	def setParams(self, sSize, aSize, l, g, eMax, eMin, eDec, activation, trainSize):
		self.sSize, self.aSize, self.g, self.eMax, self.eMin, self.eDec, self.activation, self.trainSize = sSize, aSize, g, eMax, eMin, eDec, activation, trainSize
		self.l = deque(maxlen=l)
	
	def build(self):
		#build network
		nn = Sequential([
			Dense(24, input_shape=(self.sSize,),activation=self.activation),
			Dense(12, activation=self.activation),
			Dense(12,activation=self.activation),
			Dense(2, activation="linear"),
		])
		
		#compile our model, apply optimization function
		nn.compile(loss="mse",optimizer=Adam(lr=0.001))
		
		#self.nn stores the system
		self.nn = nn
		
	def leftOrRight(self, s):
		#If our machine thinks a way is best we go that way
		if(self.eMax < random.uniform(0,1)):
			return numpy.argmax(self.nn.predict(s)[0])
		#Otherwise we choose a random direction
		else:
			return random.randint(0,1)
	
	#Save results for training purposes
	def save(self, tup):
		self.l.append(tup)
	
	#If the algorithm gets stuck we need to reset it
	def resetToTraining(self):
		tData = random.sample(self.l, self.trainSize)
		
		if self.eMin < self.eMax:
			self.eMax = self.eMax * self.eDec
			
			
		for curFrame, action, reward, nextFrame, done in tData:
			target = reward
			if (not done):
				target = reward + self.g * numpy.amax(self.nn.predict(nextFrame)[0])
			target_f = self.nn.predict(curFrame)
			target_f[0][action] = target
			self.nn.fit(curFrame, target_f, epochs=1, verbose=0)
		