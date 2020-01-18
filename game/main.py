"""
#Add these pips for colab
!apt-get install -y xvfb python-opengl > /dev/null 2>&1
!pip install gym pyvirtualdisplay > /dev/null 2>&1
"""


#Import game and dependency
import gym as game
import numpy
"""
#Import libraries to display game on colab
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
"""
#Import Keras for NN base
from keras import *

#Import custom DQN class
import sys
from deepQNetwork import deepQNetwork

#Import python libraries
import random
from collections import *

"""
#Prepare display for colab
display = Display(visible=0, size=(400, 300))
display.start()
"""

#this variable will hold the game handler
cartpole = game.make("CartPole-v1")
#Got to make sure it's clean to prevent errors
cartpole.reset()

player = deepQNetwork()
player.setParams(cartpole.observation_space.shape[0], cartpole.action_space.n, 1500, .96, 1.0, .01, .999, "relu", 30)
player.build()


#These variables are for colab-mode
flip = True
flop = True
prevs = []
oString = ""


print("How many explorations should we perform? ")
expMax = int(input())
expCur = 0

dead = False

while (expCur < expMax):
	expCur += 1
	#Start each exploration with a clean cartpole
	curFrame = cartpole.reset()
	curFrame = numpy.reshape(curFrame, [1, cartpole.observation_space.shape[0]])
	
	timeSurvived = 0
	while (not dead):
		timeSurvived = timeSurvived + 1
		cartpole.render()
		action = player.leftOrRight(curFrame)
		nextFrame, reward, dead, _ = cartpole.step(action)
		if(dead):
			reward = -10
		else:
			reward = reward
		
		nextFrame = numpy.reshape(nextFrame, [1, cartpole.observation_space.shape[0]])
		player.save((curFrame, action, reward, nextFrame, dead))
		curFrame = nextFrame
		if flip:
			prev_screen = cartpole.render(mode='rgb_array')
			#Save the frame to replay later
			prevs.append(prev_screen)
			"""
			plt.imshow(prev_screen)
			ipythondisplay.clear_output(wait=True)
			ipythondisplay.display(plt.gcf())
			"""
			flip = False
		else:
			flip = True
		
		#print results
		if dead:
			print("Exploration: " + str(expCur) + " of " + str(expMax) + " Time: " + str(timeSurvived) + " epsilon value " + str(player.eMax))
			oString += "Exploration: " + str(expCur) + " of " + str(expMax) + " Time: " + str(timeSurvived) + " epsilon value " + str(player.eMax)
			oString += "\n"
			dead = False
			break
			
	#double-back if current learning isn't working
		if len(player.l) > player.trainSize:
			player.resetToTraining()


"""
#This is your final render on colab
prev_screen = cartpole.render(mode='rgb_array')
plt.imshow(prev_screen)
ipythondisplay.clear_output(wait=True)
ipythondisplay.display(plt.gcf())


#Displays the collected output and frames for colab
print(oString)

print("Number of frames: " + str(len(prevs)))

count = 0
for i in prevs:
  print(count)
  count += 1
  plt.imshow(i)
  ipythondisplay.clear_output(wait=True)
  ipythondisplay.display(plt.gcf())
"""