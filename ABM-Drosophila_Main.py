### Agent-based model of Drosophila decision-making

## Dependencies
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib as mpl
import seaborn as sns
import random
from PIL import Image
import pandas as pd


## Define custom arctan2 function that outputs between 0 and 2pi; Output is not in Pi
def findatan2(x,y):
    arctangent2 = np.pi*(1.0-0.5*(1+np.sign(x))*(1-np.sign(y**2))-0.25*(2+np.sign(x))*np.sign(y))-np.sign(x*y)*np.arctan((np.abs(x)-np.abs(y))/(np.abs(x)+np.abs(y)))
    return float(arctangent2)

## Set up Y-maze map as binary array; size equals empirical data; each cell equals a pixel
print('Initializing Y-maze map...')
# Load picture of Y-maze outline
img = Image.open('/Users/alisc/Github/ABM-Drosophila/Ymaze.png')
# Convert image into array
imgarray = np.asarray(img)
imgYmaze = (imgarray[:,:,0] == 255).astype(bool)
# Invert Ymaze so 1s are area and 0s are out of bounds
imgYmaze = ~imgYmaze
# Transform binary Ymaze image into a coordinate array with the correctorientation
Ymaze = np.transpose( (np.rot90(np.flipud(imgYmaze == 1))).nonzero() )
# Save Ymaze bounds for later arm and related turning detection
YmazeXmax = np.shape(imgYmaze)[1]
YmazeYmax = np.shape(imgYmaze)[0]
# Determine polygons outlining arms (extends into negatives to make sure every point is considered as part of an arm)
lArmPoly = Path([(-50, 0), (YmazeXmax/2, YmazeXmax), (-50, YmazeXmax), (0,0)], closed=True)
rArmPoly = Path([(YmazeXmax+50, 0), (YmazeXmax/2, YmazeXmax), (YmazeXmax+50, YmazeXmax), (0,0)], closed=True)
bArmPoly = Path([(YmazeXmax/3, -50), (YmazeXmax/1.5, -50), (YmazeXmax/1.5, YmazeXmax/3), (YmazeXmax/3, YmazeXmax/3), (0,0)], closed=True)


## Set up speed distribution
mu = 5
sigma = 0.25
tmpSpdHist = np.histogram(np.random.normal(mu,sigma,100000), bins=np.linspace(0,mu*2,100))
# Append a zero to fix bin number discrepancy
tmp1 = np.append(np.asarray(tmpSpdHist[0]),0)
tmp2 = np.asarray(tmpSpdHist[1])
spdDist = [tmp1, tmp2]

## Initialize fly agent object w/ current position, last position, current heading angle, last heading angle, current and last speed, out of bounds counter, valid coordinates possible within the environment, flies' angular velocity bias, current and last arm turned into, current and last left or right turn made
class flyAgent:
    def __init__(self, curPos=None, lastPos=None, lastPosBackUp=None, curAngleAbs=None, curAngleRel=None, lastAngleAbs=None, lastAngleAbsBackUp=None, lastAngleRel=None, curSpd=None, lastSpd=None, OOB=None, validCoords=None, angleBias=None, curArm=None, curTurn=None):
        self.curPos = curPos if curPos is not None else np.zeros(2, dtype=int)
        self.lastPos = lastPos if lastPos is not None else np.zeros(2, dtype=int)
        self.lastPosBackUp = lastPosBackUp if lastPosBackUp is not None else np.zeros(2, dtype=int)
        self.curAngleAbs = curAngleAbs if curAngleAbs is not None else np.zeros(1, dtype=int)
        self.curAngleRel = curAngleRel if curAngleRel is not None else np.zeros(1, dtype=int)
        self.lastAngleAbs = lastAngleAbs if lastAngleAbs is not None else np.zeros(1, dtype=int)
        self.lastAngleAbsBackUp = lastAngleAbsBackUp if lastAngleAbsBackUp is not None else np.zeros(1, dtype=int)
        self.lastAngleRel = lastAngleRel if lastAngleRel is not None else np.zeros(1, dtype=int)
        self.curSpd = curSpd if curSpd is not None else np.zeros(1, dtype=int)
        self.lastSpd = lastSpd if lastSpd is not None else np.zeros(1, dtype=int)
        self.OOB = OOB if OOB is not None else np.zeros(1, dtype=int)
        self.validCoords = validCoords
        self.angleBias = angleBias if angleBias is not None else np.zeros(1, dtype=float)
        self.curArm = curArm if curArm is not None else 'spawned'
        self.curTurn = curTurn if curTurn is not None else np.nan

# Spawn fly in random valid (i.e., inside the Y-maze) starting location (matching empirical behavioral assay start)
def spawnFly(Ymaze, imgYmaze, flySpd=5, angleBias=0.5, startPos=None):

    # If starting position is not explicitly called, choose randomly based on binary map
    if startPos==None:
        startPos = list( Ymaze[ random.randint(0,len(Ymaze)-1) ] )
    fly = flyAgent()
    # Set 'last' position as random starting position
    fly.lastPos = startPos.copy()
    # Randomly (uniform) choose (absolute) heading angle at time of spawn
    fly.curAngleAbs = math.radians(random.randint(0,359))
    # Set speed to 1 pixel for heading computation
    fly.curSpd = flySpd
    # Assign current position based on valid last position and randomly chosen absolute heading direction
    fly.curPos[0] = np.round(fly.lastPos[0] + fly.curSpd * math.cos(fly.curAngleAbs))
    fly.curPos[1] = np.round(fly.lastPos[1] + fly.curSpd * math.sin(fly.curAngleAbs))
    # Assign starting relative heading direction to be 0; fly is moving straight ahead
    fly.curAngleRel = 0

    ## Set up individual angle distribution including handedness bias as vector of length 359 w/ cumulative probability at angle entries
    # Handedness bias
    fly.angleBias = angleBias
    # Mean relative angle based on individual handedness bias
    mu = 180 + ((fly.angleBias * 20) - 10)
    # Variance of angle distribution
    sigma = 20
    tmpangleDistMaster = np.histogram(np.random.normal(mu,sigma,36000000),bins=np.linspace(0,359,360))
    tmp1 = np.append(np.asarray(tmpangleDistMaster[0]),0)
    tmp2 = np.asarray(tmpangleDistMaster[1])
    fly.masterAngleDist = [tmp1, tmp2]
    # Also set up temporary angle distribution (based on master) to be updated each frame depending on environment
    fly.angleDist = fly.masterAngleDist.copy()

    # Save representation of Ymaze array in fly object for wall detection
    tmpSize = np.shape(imgYmaze)
    fly.validCoords = np.zeros([tmpSize[1], tmpSize[0]],dtype=bool)

    for cell in range(0,len(Ymaze)):
        fly.validCoords[tuple(Ymaze[cell])] = True

    return fly

# Choose a new angle for fly object at frame f based on angle distribution (and context-dependent variables v with weights wn)
def chooseAngle(fly, mu=180, sigma=10, angleDistVarInc=1):

    # Update fly object and move last frame's called angle from 'current' to 'last' parameter
    # Note that angle is heading angle relative to fly here
    fly.lastAngleRel = fly.curAngleRel
    # Randomly choose angle and convert it to fly-relative heading angle (0 or 360 are straight ahead)
    fly.curAngleRel = (np.random.normal( mu + ((fly.angleBias * 20) - 10) , sigma + (fly.OOB * angleDistVarInc) ) + 180)
    if fly.curAngleRel >= 360:
        fly.curAngleRel = abs(fly.curAngleRel - 360)

    # Convert relative angle to radians
    fly.curAngleRel = math.radians(fly.curAngleRel)

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

    # Determine last absolute heading direction
    fly.lastAngleAbsBackUp = fly.lastAngleAbs
    #fly.lastAngleAbs = findatan2( fly.curPos[0] - fly.lastPos[0], fly.curPos[1] - fly.lastPos[1] )
    fly.lastAngleAbs = fly.curAngleAbs
    # If last and current position are the same, last absolute angle is NaN; reassign old heading angle
    if np.isnan(fly.lastAngleAbs):
        fly.lastAngleAbs = fly.lastAngleAbsBackUp
    # Update current absolute heading direction based on last absolute heading direction and current relative heading
    fly.curAngleAbs = fly.lastAngleAbs + fly.curAngleRel

    # Set previous current position as last position for purpose of calculating the next position
    fly.lastPosBackUp = fly.lastPos.copy()
    fly.lastPos = fly.curPos.copy()

    # Make sure that fly-relative heading angle is already converted to map-relative cardinal angle
    fly.curPos[0] = fly.lastPos[0] + fly.curSpd * math.cos(fly.curAngleAbs)
    fly.curPos[1] = fly.lastPos[1] + fly.curSpd * math.sin(fly.curAngleAbs)
    print('Proposing position:', fly.curPos)

    # Check if current position is NOT out of bounds of the Ymaze environment
    if not any(fly.curPos >= np.shape(fly.validCoords)) and not any(fly.curPos < 0):
        # Check if current position is NOT within the maze itself
        if not fly.validCoords[ fly.curPos[0], fly.curPos[1] ]:
            # If position is not valid coordinate, reassign old position as current position
            fly.curPos = fly.lastPos.copy()
            fly.lastPos = fly.lastPosBackUp.copy()
            fly.curAngleAbs = fly.lastAngleAbs
            fly.lastAngleAbs = fly.lastAngleAbsBackUp
            # Also report that fly would have been out of bounds
            fly.OOB += 1
            print('Current position hit a wall, resetting to', fly.curPos)
        else:
            # Valid position, proceed as normal and reset out of bounds counter
            fly.OOB = 0
            print('Accepted proposed position ', fly.curPos)
    else:
        # If out of bounds of Ymaze environment, reassign old position as current position
        fly.curPos = fly.lastPos.copy()
        fly.lastPos = fly.lastPosBackUp.copy()
        fly.curAngleAbs = fly.lastAngleAbs
        fly.lastAngleAbs = fly.lastAngleAbsBackUp
        # Also report that fly would have been out of bounds
        fly.OOB += 1
        print('Proposed position is out of bounds, resetting to', fly.curPos)
        
    return fly

def updateTurn(fly, bArmPoly, lArmPoly, rArmPoly):
    # If current position was valid
    if fly.OOB == 0:
        # If last arm was bottom arm, check if current position is in either of the other two
        if fly.curArm == 'b':
            if lArmPoly.contains_point(fly.curPos):
                fly.curTurn = 0
                fly.curArm = 'l'
            elif rArmPoly.contains_point(fly.curPos):
                fly.curTurn = 1
                fly.curArm = 'r'
            # If current position between arms or still in current arm, set current turn to None but maintain current arm
            else:
                fly.curTurn = np.nan
        # If last arm was left arm, check the other two
        elif fly.curArm == 'l':
            if rArmPoly.contains_point(fly.curPos):
                fly.curTurn = 0
                fly.curArm = 'r'
            elif bArmPoly.contains_point(fly.curPos):
                fly.curTurn = 1
                fly.curArm ='b'
            else:
                fly.curTurn = np.nan
        # If last arm was right arm
        elif fly.curArm == 'r':
            if bArmPoly.contains_point(fly.curPos):
                fly.curTurn = 0
                fly.curArm = 'b'
            elif lArmPoly.contains_point(fly.curPos):
                fly.curTurn = 1
                fly.curArm = 'l'
            else:
                fly.curTurn = np.nan
        # If current arm is none fly was just spawned, assign current arm correctly and omit a turn from being scored
        elif fly.curArm == 'spawned':
            if bArmPoly.contains_point(fly.curPos):
                fly.curArm = 'b'
            elif lArmPoly.contains_point(fly.curPos):
                fly.curArm = 'l'
            elif rArmPoly.contains_point(fly.curPos):
                fly.curArm = 'r'

    return fly

## Run, save, and visualize a fly experiment
# Expmt is an array with N of duration rows
# Expmt columns are X[0], Y[1], current Turn number [2], curent Turn direction (left: 0, right:1) [3], current absolute heading angle [4], current relative angular velocity angle [5] 
def runExperiment(Ymaze, imgYmaze, bArmPoly, lArmPoly, rArmPoly, duration, flySpd, angleBias, visualize=False):
    fly = spawnFly(Ymaze, imgYmaze, flySpd, angleBias)
    # Set up experimental data array
    expmt = np.zeros([duration, 6])
    # Change turn zeros to NaNs
    expmt[:,:].fill(np.nan)

    # Set up temporary turn counter and frame counter
    tempturn = 0
    frame = 0

    while frame < duration:
        chooseAngle(fly)
        updatePos(fly)
        updateTurn(fly, bArmPoly, lArmPoly, rArmPoly)
        expmt[frame,0] = fly.curPos[0].copy()
        expmt[frame,1] = fly.curPos[1].copy()
        tempturn = tempturn + ~np.isnan(fly.curTurn).copy()
        expmt[frame,2] = tempturn.copy()
        expmt[frame,3] = fly.curTurn
        expmt[frame,4] = fly.curAngleAbs
        expmt[frame,5] = fly.curAngleRel

        if fly.OOB == 0:
            frame += 1

    # Compute summary statistics for simulated fly
    if ~np.isnan( sum((expmt[:,3])) ):
        fly.rBias = sum( expmt[~np.isnan(expmt[:,3]),3] ) / len(expmt[~np.isnan(expmt[:,3])]) 

    if visualize:
        plt.scatter( Ymaze[:,0], Ymaze[:,1],color='red' )
        plt.plot( expmt[:,0], expmt[:,1] )

    return expmt, fly