### Agent-based model of Drosophila decision-making

## Dependencies
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from PIL import Image

## Set up Y-maze map as binary array; size equals empirical data; each cell equals a pixel
print('Initializing Y-maze map...')

# Load picture of Y-maze outline
img = Image.open('/Users/alisc/Github/ABM-Drosophila/Ymaze.png')
# Convert image into array
imgarray = np.asarray(img)
Ymaze = (imgarray[:,:,0] == 255).astype(bool)

sns.heatmap(Ymaze)

## Set up (empirical) master angle distribution as vector of length 359 w/ cumulative probability at angle entries
# mean relative angle
mu = 180
# variance of angle distribution
sigma = 10

angleDistMaster = tuple(np.round(np.histogram(np.random.normal(mu,sigma,36000000),359)[0]))

# Also set up temporary angle distribution (based on master) to be updated each frame
angleDist = angleDistMaster


## Initialize fly agent object w/ current position, last position, current heading angle
class flyAgent:
    def __init__(self, curPos=np.zeros(2, dtype=int), lastPos=np.zeros(2, dtype=int), curAngle=np.zeros(1, dtype=int), lastAngle=np.zeros(1, dtype=int), curSpd=np.zeros(1, dtype=int), lastSpd=np.zeros(1, dtype=int)):
        self.curPos = curPos       
        self.lastPos = lastPos
        self.curAngle = curAngle
        self.lastAngle = lastAngle
        self.curSpd = curSpd
        self.lastSpd = lastSpd

# Spawn fly in random valid (i.e., inside the Y-maze) starting location (matching empirical behavioral assay start)
def spawnFly(Ymaze, startPos=None):
    validPositions = np.transpose( (Ymaze == 1).nonzero() )
    # if starting position is not explicitly called, choose randomly based on binary map
    if startPos==None:
        startPos = list( validPositions[ random.randint(0,len(validPositions)-1) ] )
    fly = flyAgent()
    fly.curPos = startPos
    return fly

# Update angle distribution based on prior (master, empirical angle distribution) and current context (i.e., angles leading to impossible locations)
def updateAngleDist(fly, angleDist):

    return angleDist


# Choose a new angle for fly object at frame f based on angle distribution (and context-dependent variables v with weights wn)
def chooseAngle(fly, angleDist):
    # Update fly object and move last frame's called angle from 'current' to 'last' parameter
    # Note that angle is heading angle relative to fly here
    fly.lastAngle = fly.curAngle
    fly.curAngle = random.choices( range(0,359), angleDist )

    return fly


# Choose speed based on previously chosen new angle (speed empirically depends on angle)
def chooseSpd(fly, spdDist):
    # Set just utilized speed as last speed
    fly.lastSpd = fly.curSpd
    # Choose new speed from speed distributions relative to current angle heading
    return fly

# Update new fly position based on chosen angle and speed
def updatePos(fly):
    # Set previous current position as last position for purpose of calculating the next position
    fly.lastPos = fly.curPos
    # Convert fly-relative heading angle to map-relative cardinal angle
    fly.curPos[0] = np.round(fly.lastPos[0] + fly.curSpd * math.cos(math.radians(fly.curAngle[0])))
    fly.curPos[1] = np.round(fly.lastPos[1] + fly.curSpd * math.sin(math.radians(fly.curAngle[0])))
    


    return fly