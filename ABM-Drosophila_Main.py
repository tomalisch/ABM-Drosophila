### Agent-based model of Drosophila decision-making

## Dependencies
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib as mpl
import seaborn as sns
import random
from PIL import Image
import pandas as pd
import scipy.io
import datetime
import hdf5storage
from tqdm.auto import tqdm


## Define function that returns occupied fly coordinates by body size, given a centroid position (fly shape assumed circular)
def circleCoords(r, x0, y0 ):

    x_ = np.arange(x0 - r - 1, x0 + r + 1, dtype=int)
    y_ = np.arange(y0 - r - 1, y0 + r + 1, dtype=int)
    x, y = np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 <= r**2)
    cCoords = []
    for x, y in zip(x_[x], y_[y]):
        cCoords.append([x, y])
    return np.asarray(cCoords)
 
 
## Define function that returns absolute angle in radians between two consecutive points
# Note that p1 is the first point and p2 is the second point
# Output is in radians from (0,2pi)
def getAngleAbs(p1, p2):

    angleAbs = np.mod( math.atan2(p2[1]-p1[1] , p2[0]-p1[0]), 2*np.pi)
    return angleAbs


## Set up Y-maze map as binary array; size equals empirical data; each cell equals a pixel
# Load picture of Y-maze outline
img = Image.open('/Users/alisc/Github/ABM-Drosophila/Ymaze.png')
# Convert image into array
imgarray = np.asarray(img)
imgYmaze = (imgarray[:,:,0] == 255).astype(bool)
# Invert Ymaze so 1s are area and 0s are walls/out of bounds
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

## Initialize fly agent object w/ current position, last position, current heading angle, last heading angle, current and last speed, out of bounds counter, valid coordinates possible within the environment, flies' angular velocity bias, current and last arm turned into, current and last left or right turn made, and body size (in px radius, default is 2=13 total pixels)
class flyAgent:
    def __init__(self, curPos=None, lastPos=None, lastPosBackUp=None, curAngleAbs=None, curAngleRel=None, lastAngleAbs=None, lastAngleAbsBackUp=None, lastAngleRel=None, curSpd=None, lastSpd=None, OOB=None, validCoords=None, angleBias=None, curArm=None, curTurn=None, nTurn=None, rBias=None, seqEff=None, bodySize=None, startArmTurn=None):
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
        self.nTurn = nTurn if nTurn is not None else np.zeros(1, dtype=int)
        self.rBias = rBias if rBias is not None else np.nan
        self.seqEff = seqEff if seqEff is not None else np.nan
        self.bodySize = bodySize if bodySize is not None else np.zeros(1, dtype=int)
        self.startArmTurn = startArmTurn if startArmTurn is not None else np.nan

# Spawn fly in random valid (i.e., inside the Y-maze) starting location (matching empirical behavioral assay start)
def spawnFly(Ymaze, imgYmaze, flySpd=5, angleBias=0.5, startPos=None, bodySize=2):

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

    # Set fly body size
    fly.bodySize = bodySize

    # Handedness bias
    fly.angleBias = angleBias

    # Save representation of Ymaze array in fly object for wall detection
    tmpSize = np.shape(imgYmaze)
    fly.validCoords = np.zeros([tmpSize[1], tmpSize[0]],dtype=bool)
    for cell in range(0,len(Ymaze)):
        fly.validCoords[tuple(Ymaze[cell])] = True

    return fly

# Choose a new angle for fly object at frame f based on angle distribution (and context-dependent variables v with weights wn)
def chooseAngle(fly, mu=180, sigma=10, angleDistVarInc=0.1, brownMotion=False):

    # Update fly object and move last frame's called angle from 'current' to 'last' parameter
    # Note that angle is heading angle relative to fly here
    fly.lastAngleRel = fly.curAngleRel
    # Randomly choose angle and convert it to fly-relative heading angle (0 or 360 are straight ahead)
    # If brownian motion mode, take a random integer heading
    if brownMotion:
        fly.curAngleRel = random.randint(0,359)
    # If not brownian motion, take a random heading from the distribution centered on fly's angleBias with defined sigma
    else:
        fly.curAngleRel = np.random.normal( mu + ((fly.angleBias * 20) - 10) , sigma + (fly.OOB * angleDistVarInc) ) + 180
        if fly.curAngleRel >= 360:
            fly.curAngleRel = abs(fly.curAngleRel - 360)

    # Convert relative angle to radians
    fly.curAngleRel = math.radians(fly.curAngleRel)

    return fly

# Detect whether a wall is in range and returns angle and distance of closest wall to fly. Radius is in times bodysize.
def detectWall(fly, detectRadius=1.5):

    # Use circleCoords to draw sensory boundary around centroid at current location
    detectCoords = circleCoords(fly.bodySize * detectRadius, fly.curPos[0], fly.curPos[1])

    # Return array of coordinates within detectCoords that are 0 w/ respect to global valid coordinates (detect a wall)
    wallCoords = detectCoords[ fly.validCoords[ detectCoords[:,0], detectCoords[:,1] ] == 0 ]

    # If exactly one wall coordinate was found in detection radius:
    if len(wallCoords) == 1:
        # Determine shortest wall distance from fly centroid and calculate absolute angle of that point
        wallDistances = np.linalg.norm(wallCoords - fly.curPos)
        wallCoordsMinDist = wallCoords[ wallDistances==np.min(wallDistances)][0]
        wallAngle = getAngleAbs(fly.curPos, wallCoordsMinDist)
        wallDist = np.linalg.norm(wallCoordsMinDist - fly.curPos)

    # If more than one wall coordinate was detected
    elif len(wallCoords) > 1:
        # If more than one coordinate found, iterate over resulting angles and choose point that results in the angle closest to current absolute heading angle
        wallDistances = np.linalg.norm(wallCoords - fly.curPos, axis=1)
        wallCoordsMinDist = wallCoords[ wallDistances==np.min(wallDistances)][0]
        wallAngle = getAngleAbs(fly.curPos, wallCoordsMinDist)
        wallDist = np.linalg.norm(wallCoordsMinDist - fly.curPos)

    # If no wall was in detection range
    else:
        wallAngle = None
        wallCoordsMinDist = None
        wallDist = None

    return wallAngle, wallDist, wallCoordsMinDist


# Choose speed based on previously chosen new angle (speed empirically depends on angle)
def chooseSpd(fly, mu=5, sigma=1, spdVarInc=0.1):

    # Set just utilized speed as last speed
    fly.lastSpd = fly.curSpd
    # Choose new speed from speed distributions relative to current angle heading
    # Note: Currently sampling from Gaussian
    fly.curSpd = np.random.normal( mu, sigma + (fly.OOB * spdVarInc) )

    return fly

# Update new fly position based on chosen angle and speed
def updatePos(fly, wallFollowing=True, wallBias=0.5, detectRadius=1.5):

    # Determine last absolute heading direction
    fly.lastAngleAbsBackUp = fly.lastAngleAbs
    #fly.lastAngleAbs = findatan2( fly.curPos[0] - fly.lastPos[0], fly.curPos[1] - fly.lastPos[1] )
    fly.lastAngleAbs = fly.curAngleAbs
    # If last and current position are the same, last absolute angle is NaN; reassign old heading angle
    if np.isnan(fly.lastAngleAbs):
        fly.lastAngleAbs = fly.lastAngleAbsBackUp

    # If fly should wall follow, adjust mean of heading direction distribution to be mixed between straight ahead & closest wall angle (weighted by wallBias)
    # Note that wall 'attraction' increases as distance to wall decreases
    if wallFollowing:
        wallAngle, wallDist, wallCoordsMinDist = detectWall(fly, detectRadius=detectRadius)
        # Take the wallBias weighted mean of wallAngle and last heading angle to update proposed angle before adding relative heading angle
        fly.lastAngleAbs = ( (wallBias * wallAngle) + ((1 - wallBias) * fly.lastAngleAbs) ) / 2 

    # Update current absolute heading direction based on last absolute heading direction and current relative heading
    fly.curAngleAbs = ((fly.lastAngleAbs + fly.curAngleRel) % (2*math.pi))

    # Set previous current position as last position for purpose of calculating the next position
    fly.lastPosBackUp = fly.lastPos.copy()
    fly.lastPos = fly.curPos.copy()

    # Make sure that fly-relative heading angle is already converted to map-relative cardinal angle
    fly.curPos[0] = fly.lastPos[0] + fly.curSpd * math.cos(fly.curAngleAbs)
    fly.curPos[1] = fly.lastPos[1] + fly.curSpd * math.sin(fly.curAngleAbs)
    #print('Proposing position:', fly.curPos)

    # Account for fly size for out-of-bounds checking with temporary variable
    flyBodyCoords = circleCoords(fly.bodySize, fly.curPos[0], fly.curPos[1])

    # Check if any body-size-dependent proposed fly positions is out of bounds of the Ymaze environment or inside a wall
    if np.any(flyBodyCoords < 0) or np.any(flyBodyCoords >= np.asarray([YmazeXmax, YmazeYmax])):
        # Fly outside of maze array bounds, reset porposed position and return early
        fly.curPos = fly.lastPos.copy()
        fly.lastPos = fly.lastPosBackUp.copy()
        fly.curAngleAbs = fly.lastAngleAbs
        fly.lastAngleAbs = fly.lastAngleAbsBackUp
        # Also report that fly would have been out of bounds
        fly.OOB += 1
        # Return early
        # print('out of bounds at ', fly.curPos)
        return fly
    
    # Check if proposed position would cause fly to be inside a wall
    if not all( fly.validCoords[ flyBodyCoords[:,0], flyBodyCoords[:,1] ]):
        # Proposed position would be inside wall, reassign old position as current position
        fly.curPos = fly.lastPos.copy()
        fly.lastPos = fly.lastPosBackUp.copy()
        fly.curAngleAbs = fly.lastAngleAbs
        fly.lastAngleAbs = fly.lastAngleAbsBackUp
        fly.OOB += 1
        # print('Hit wall at ', fly.curPos)
        return fly
    
    # If both checks passed: Valid position, proceed as normal and reset out of bounds counter
    fly.OOB = 0
    #print('Accepted proposed position ', fly.curPos)   
    return fly


# Update and record whether a turn was mcompleted/started on the current frame
# curArm tracks arm the fly is currently in and is used for computing the other metrics
# curTurn is nan if no turn was completed on current frame, 0 for left turn, 1 for right turn
# startArmTurn is 0 if the completed turn started from the bottom arm, 1 if started from left arm, 2 if started from right

def updateTurn(fly, bArmPoly, lArmPoly, rArmPoly):
    # If current position was valid
    if fly.OOB == 0:
        # If last arm was bottom arm, check if current position is in either of the other two
        if fly.curArm == 'b':
            if lArmPoly.contains_point(fly.curPos):
                fly.curTurn = 0
                fly.nTurn += 1
                fly.curArm = 'l'
                fly.startArmTurn = 0
            elif rArmPoly.contains_point(fly.curPos):
                fly.curTurn = 1
                fly.nTurn += 1
                fly.curArm = 'r'
                fly.startArmTurn = 0
            # If current position between arms or still in current arm, set current turn to None but maintain current arm
            else:
                fly.curTurn = np.nan
        # If last arm was left arm, check the other two
        elif fly.curArm == 'l':
            if rArmPoly.contains_point(fly.curPos):
                fly.curTurn = 0
                fly.nTurn += 1
                fly.curArm = 'r'
                fly.startArmTurn = 1
            elif bArmPoly.contains_point(fly.curPos):
                fly.curTurn = 1
                fly.nTurn += 1
                fly.curArm ='b'
                fly.startArmTurn = 1
            else:
                fly.curTurn = np.nan
        # If last arm was right arm
        elif fly.curArm == 'r':
            if bArmPoly.contains_point(fly.curPos):
                fly.curTurn = 0
                fly.nTurn += 1
                fly.curArm = 'b'
                fly.startArmTurn = 2
            elif lArmPoly.contains_point(fly.curPos):
                fly.curTurn = 1
                fly.nTurn += 1
                fly.curArm = 'l'
                fly.startArmTurn = 2
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
# Expmt columns are X[0], Y[1], current Turn number [2], curent Turn direction (left: 0, right:1) [3], current Turn arm start (0: bottom arm, 1: left, 2: right) [4], current absolute heading angle [5], current relative angular velocity angle [6] 
def assayFly(Ymaze, imgYmaze, bArmPoly, lArmPoly, rArmPoly, duration, flySpd, angleBias, av_sigma, bodySize, brownMotion, wallFollowing, wallBias, detectRadius):

    fly = spawnFly(Ymaze, imgYmaze, flySpd=flySpd, angleBias=angleBias, startPos=None, bodySize=bodySize)
    # Set up experimental data array
    expmt = np.zeros([duration, 7])
    # Change turn zeros to NaNs
    expmt[:,:].fill(np.nan)

    # Set up temporary experimental frame and total simulation cycle counters
    frame = 0
    cycle = 0

    while frame < duration:
        chooseAngle(fly, sigma=av_sigma, brownMotion=brownMotion)
        updatePos(fly, wallFollowing=wallFollowing, wallBias=wallBias, detectRadius=detectRadius)
        updateTurn(fly, bArmPoly, lArmPoly, rArmPoly)
        expmt[frame,0] = fly.curPos[0].copy()
        expmt[frame,1] = fly.curPos[1].copy()
        expmt[frame,2] = fly.nTurn
        expmt[frame,3] = fly.curTurn
        expmt[frame,4] = fly.startArmTurn
        expmt[frame,5] = fly.curAngleAbs
        expmt[frame,6] = fly.curAngleRel

        # Keep track of overall cycles to read out if fly gets 'stuck'
        cycle += 1

        # If proposed position was accepted, advance frame
        if fly.OOB == 0:
            frame += 1

        # If cycles exceed duration frames by 3 orders of magnitude, break out of while loop. Un-simulated frames will remain NaNs in expmt output
        if cycle >= duration*1000:
            print('ABM exceeded cycle allowance. Check output for NaNs.')

            return expmt, fly

    ## Compute summary statistics for simulated fly
    # Compute sequential effect of choosing right turn given last turn decision was right turn as well
    if np.nansum((expmt[:,3])) != 0:
        fly.rBias = np.nansum( expmt[~np.isnan(expmt[:,3]),3] ) / len(expmt[~np.isnan(expmt[:,3])])
        turnseq = list(enumerate(expmt[~np.isnan(expmt[:,3]),3]))
        # If fly made more than 1 turn
        if len(turnseq) > 1:
            rrseqCounter = 0
            llseqCounter = 0
            for i in range(1,len(turnseq)):
                rrseqCounter += (turnseq[:][i][1] * turnseq[:][i-1][1])
                llseqCounter += (turnseq[:][i][1] + turnseq[:][i-1][1])==0
            fly.seqEff = rrseqCounter/(len(turnseq)-1)
        else:
        # If fly made 1 or fewer turns, set seqEff to NaN
            fly.seqEff = np.nan
    return expmt, fly


def runExperiment(flyN, Ymaze, imgYmaze, bArmPoly, lArmPoly, rArmPoly, data = None, duration=30*60*60, flySpd=5, angleBias=0.5,  av_sigma=10, bodySize=2, visualize=False, cleanup=True, brownMotion=False, wallFollowing=True, wallBias=0.5, detectRadius=1.5):

    # If data array not defined, create it
    if data is None:
        data = np.zeros([flyN, duration, 7])

    for flyID in tqdm(range(flyN), desc='Experiment progress', leave=True):
        expmt1, fly1 = assayFly(Ymaze, imgYmaze, bArmPoly, lArmPoly, rArmPoly, duration=duration, flySpd=flySpd, angleBias=angleBias, av_sigma=av_sigma, bodySize=bodySize, brownMotion=brownMotion, wallFollowing=wallFollowing, wallBias=wallBias, detectRadius=detectRadius)

        data[flyID, :, :] = expmt1

        if cleanup:
            # Clean up memory
            del(expmt1, fly1)

    if cleanup:
        return data
    else:
        return data, expmt1, fly1
    

# Function to save data array as .mat file  
def saveExperiment(data, filename):

    # Check if filename is a string and includes '.mat' at the end; else create failsafe filename
    if not isinstance(filename, str):
        filename = 'dataABM_' + str(datetime.date.today())
    if not filename.endswith('.mat'):
        filename = filename + '.mat'

    # MATLAB field will just be 'data' to allow loading into workspace
    hdf5storage.write({'data':data}, '.', filename, matlab_compatible=True, store_python_metadata=True, compress=True)
    print(f'Saved data as {filename} in {os.getcwd()}/')
    