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
tmpangleDistMaster = np.histogram(np.random.normal(mu,sigma,36000000),bins=np.linspace(0,359,360))
tmp1 = np.append(np.asarray(tmpangleDistMaster[0]),0)
tmp2 = np.asarray(tmpangleDistMaster[1])
angleDistMaster = [tmp1, tmp2]
# Also set up temporary angle distribution (based on master) to be updated each frame depending on environment
angleDist = angleDistMaster

## Set up speed distribution
mu = 2
sigma = 0.25
tmpSpdHist = np.histogram(np.random.normal(mu,sigma,100000), bins=np.linspace(0,5,100))
# Append a zero to fix bin number discrepancy
tmp1 = np.append(np.asarray(tmpSpdHist[0]),0)
tmp2 = np.asarray(tmpSpdHist[1])
spdDist = [tmp1, tmp2]

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
    fly.lastPos = startPos
    # Randomly (uniform) choose (absolute) heading angle at time of spawn
    fly.curAngle = random.randint(0,359)
    # Set speed to 1 pixel for heading computation
    fly.curSpd = 1
    # Assign current position based on valid last position and randomly chosen absolute heading direction
    fly.curPos[0] = np.round(fly.lastPos[0] + fly.curSpd * math.cos(math.radians(fly.curAngle[0])))
    fly.curPos[1] = np.round(fly.lastPos[1] + fly.curSpd * math.sin(math.radians(fly.curAngle[0])))
    return fly

# Update angle distribution based on prior (master, empirical angle distribution) and current context (i.e., angles leading to impossible locations)
def updateAngleDist(fly, angleDist):

    return angleDist


# Choose a new angle for fly object at frame f based on angle distribution (and context-dependent variables v with weights wn)
def chooseAngle(fly, angleDist):
    # Update fly object and move last frame's called angle from 'current' to 'last' parameter
    # Note that angle is heading angle relative to fly here
    fly.lastAngle = fly.curAngle
    fly.curAngle = random.choices( angleDist[1], angleDist[0] )[0] + 180
    if fly.curAngle >= 360:
        fly.curAngle = abs(fly.curAngle - 360)

    return fly

# Convert fly-relative angle to absolute heading angle
def convertAngle(fly):
    relAngle = fly.curAngle

    return fly


# Choose speed based on previously chosen new angle (speed empirically depends on angle)
def chooseSpd(fly, spdDist):
    # Set just utilized speed as last speed
    fly.lastSpd = fly.curSpd
    # Choose new speed from speed distributions relative to current angle heading
    # Note: Currently sampling from Gaussian
    fly.curSpd = random.choices( spdDist[1],  spdDist[0] )[0]
    return fly

# Update new fly position based on chosen angle and speed
def updatePos(fly):
    # Set previous current position as last position for purpose of calculating the next position
    fly.lastPos = fly.curPos
    # Make sure that fly-relative heading angle is already converted to map-relative cardinal angle
    fly.curPos[0] = np.round(fly.lastPos[0] + fly.curSpd * math.cos(math.radians(fly.curAngle[0])))
    fly.curPos[1] = np.round(fly.lastPos[1] + fly.curSpd * math.sin(math.radians(fly.curAngle[0])))
    


    return fly