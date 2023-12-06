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


# Define function that returns occupied fly coordinates by body size, given a centroid position (fly shape assumed circular)
def circleCoords(r, x0, y0 ):

    x_ = np.arange( np.ceil(x0 - r - 1), np.ceil(x0 + r + 1), dtype=int )
    y_ = np.arange( np.ceil(y0 - r - 1), np.ceil(y0 + r + 1), dtype=int )
    x, y = np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 <= r**2)
    cCoords = []
    for x, y in zip(x_[x], y_[y]):
        cCoords.append([x, y])

    return np.asarray(cCoords)
 
# Define function that returns absolute angle in radians between two consecutive points (w/ respect to X-axis)
# Note that p1 is the first point and p2 is the second point
# Output is in radians from (0,2pi)
def getAngleAbs(p1, p2):

    angleAbs = np.mod( math.atan2(p2[1]-p1[1] , p2[0]-p1[0]), 2*np.pi)

    return angleAbs

# Compute difference of two (radians) angles
def getAngleDiff(ang1, ang2):

    angleDiff = 180 - abs(abs(np.rad2deg(ang1) - np.rad2deg(ang2)) - 180)

    return angleDiff

# Define function that returns the weighted mean angle based on relative quadrants and similarity
# Note that angleWeight applies to first angle, second angle weight is 1-angleWeight
def getWeightedAngleMean(ang1, ang2, angleWeight):

    # Take the average smallest angle (from -pi to pi)
    wmAngle = np.arctan2( (np.sin(ang1)*angleWeight)  +  (np.sin(ang2)*(1-angleWeight)), (np.cos(ang1)*angleWeight)  +  (np.cos(ang2)*(1-angleWeight)) )

    # Change angle back into (0 to 2pi) space
    if wmAngle<0:
        wmAngle = (math.pi + (wmAngle/2) ) * 2

    return wmAngle


## Set up Y-maze map as binary array; size equals empirical data; each cell equals a pixel
# Load picture of Y-maze outline
# Note that real Y-maze arms are 3.6mm wide; 120px here; 0.03px/mm
# Real Drosophila are ~2mm wide, 60px here to match arena's ~0.03px/mm ratio
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
def spawnFly(Ymaze, imgYmaze, flySpd=5, angleBias=0.5, startPos=None, bodySize=30):

    # Assign class
    fly = flyAgent()

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

    # Adjust validCoords to account for bodySize (some coordinates are not valid because fly body would extend into wall)
    # Iterate over all coordinates once and create boolean mask of actually valid coordinates
    if fly.bodySize > 1:
        for x in range(fly.bodySize,len(fly.validCoords[:,0])):
            for y in range(fly.bodySize,len(fly.validCoords[0,:])):
                tmpCoords = circleCoords(fly.bodySize, x, y)
                tmpValid = np.zeros(0)
                for tmp in range(0,len(tmpCoords[:,0])-1):
                    tmpValid = np.append(tmpValid, fly.validCoords[tuple(tuple(tmpCoords)[tmp])])
                fly.validCoords[x,y] = np.all(tmpValid)



    # If starting position is not explicitly called, choose randomly based on binary map
    # If body size is set, repeat procedure until viable spot is found
    if startPos==None:
        startPos = list( Ymaze[ random.randint(0,len(Ymaze)-1) ] )
        # Determine occupied coordinates based on starting position
        flyBodyCoords = circleCoords(bodySize, startPos[0], startPos[1])
        # While fly body would be out of bounds, repeat starting position determination
        while (np.any(flyBodyCoords < 0) or np.any(flyBodyCoords >= np.asarray([YmazeXmax, YmazeYmax]))) or (not all( fly.validCoords[ flyBodyCoords[:,0], flyBodyCoords[:,1] ])):
            startPos = list( Ymaze[ random.randint(0,len(Ymaze)-1) ] )
            flyBodyCoords = circleCoords(bodySize, startPos[0], startPos[1])

    # Set 'last' position as random starting position
    fly.lastPos = startPos.copy()

    # Randomly (uniform) choose (absolute) heading angle at time of spawn
    fly.curAngleAbs = math.radians(random.randint(0,359))

    # Assign last absolute angle to be equal to randomly chosen current one (to prevent 0 angle for wall angle detection)
    fly.lastAngleAbs = fly.curAngleAbs

    # Set speed to 1 pixel for heading computation
    fly.curSpd = flySpd

    # Assign current position based on valid last position and randomly chosen absolute heading direction
    fly.curPos[0] = np.round(fly.lastPos[0] + fly.curSpd * math.cos(fly.curAngleAbs))
    fly.curPos[1] = np.round(fly.lastPos[1] + fly.curSpd * math.sin(fly.curAngleAbs))

    return fly

# Check whether all entries in an array of coordinates are within bounds
# Note that coordArray is an array of shape (c,2) with c number of x,y coordinates
# Note that validArray is a boolean array with row, columns reflecting coordinate IDs /re coordArray
def isCoordValid(coordArray, validArray):

    # Return 1 if invalid, bool invert for nomenclature's sake
    return not ( (np.any(coordArray < 0) or np.any(coordArray >= np.asarray([np.shape(validArray)[0], np.shape(validArray)[1]]))) or (not all( validArray[ coordArray[:,0], coordArray[:,1] ])) )

# Choose a new angle for fly object at frame f based on angle distribution (and context-dependent variables v with weights wn)
def chooseAngle(fly, mu=180, av_sigma=10, angleDistVarInc=10, brownMotion=False):

    # Update fly object and move last frame's called angle from 'current' to 'last' parameter
    # Note that angle is heading angle relative to fly here
    fly.lastAngleRel = fly.curAngleRel
    # Randomly choose angle and convert it to fly-relative heading angle (0 or 360 are straight ahead)
    # If brownian motion mode, take a random integer heading
    if brownMotion:
        fly.curAngleRel = random.randint(0,359)
    # If not brownian motion, take a random heading from the distribution centered on fly's angleBias with defined sigma
    else:
        fly.curAngleRel = np.random.normal( mu + ((fly.angleBias * 20) - 10) , av_sigma + (fly.OOB * angleDistVarInc) ) + 180
        if fly.curAngleRel >= 360:
            fly.curAngleRel = abs(fly.curAngleRel - 360)

    # Convert relative angle to radians
    fly.curAngleRel = math.radians(np.squeeze(fly.curAngleRel))

    return fly

# Detect whether a wall is in range and returns angle and distance of closest wall to fly. Radius is in times bodysize.
def detectWall(fly, detectRadius=1.5):

    # Use circleCoords to draw sensory boundary around centroid at current location
    detectCoords = circleCoords(fly.bodySize * detectRadius, fly.curPos[0], fly.curPos[1])

    # Prune coordinates that extend outside the arena
    detectCoords_pruned = detectCoords[ np.array(detectCoords[:,0] < fly.validCoords.shape[0]) * np.array(detectCoords[:,1] < fly.validCoords.shape[1]) * np.array( detectCoords[:,0] > 0 ) * np.array( detectCoords[:,1] > 0) ]

    # Return array of coordinates within detectCoords that are 0 w/ respect to global valid coordinates (detect a wall)
    wallCoords = detectCoords_pruned[ fly.validCoords[ detectCoords_pruned[:,0], detectCoords_pruned[:,1] ] == 0 ]

    # If exactly one wall coordinate was found in detection radius:
    if len(wallCoords) == 1:
        # Determine shortest wall distance from fly centroid and calculate absolute angle of that point
        wallDistances = np.linalg.norm(wallCoords - fly.curPos)
        wallCoordsMinDist = np.squeeze(wallCoords[ wallDistances==np.min(wallDistances)])
        wallAngle = getAngleAbs(fly.curPos, wallCoordsMinDist)
        wallDist = np.linalg.norm(wallCoordsMinDist - fly.curPos)

    # If more than one wall coordinate was detected
    elif len(wallCoords) > 1:
        # If more than one coordinate found, iterate over resulting angles and choose point that results in the angle closest to current absolute heading angle
        wallAngles = np.zeros(len(wallCoords))
        angleDiff = np.zeros(len(wallCoords))
        wallDistances = np.linalg.norm(wallCoords - fly.curPos, axis=1)
        # Iterate through wall coordinates and find those with lowest angle difference to current absolute heading direction
        for wd in range(0,len(wallDistances)):
            wallAngles[wd] = getAngleAbs(fly.curPos, wallCoords[wd])
            angleDiff[wd] = getAngleDiff(fly.lastAngleAbs, wallAngles[wd])

        # Assign returned wallAngle as the angle with the smallest difference to current heading direction (If true for multiple angles, take the first one)
        minAngleDiffID = np.where(angleDiff==min(angleDiff))[0]
        wallAngle = wallAngles[ minAngleDiffID ][0]
        # Same for wall coordinate distance
        wallDist = wallDistances[ minAngleDiffID][0]

    # If no wall was in detection range
    else:
        wallAngle = None
        wallCoordsMinDist = None
        wallDist = None
        return wallAngle, wallDist

    return wallAngle, wallDist

# Detect accessible coordinates around the fly, and their respective angles and distances
def detectOpenCoords(fly, detectRadius=1.5):

    # Use circleCoords to draw sensory boundary around centroid at current location
    detectCoords = circleCoords(fly.bodySize * detectRadius, fly.curPos[0], fly.curPos[1])

    # Prune coordinates that extend outside the arena
    detectCoords_pruned = detectCoords[ np.array(detectCoords[:,0] < fly.validCoords.shape[0]) * np.array(detectCoords[:,1] < fly.validCoords.shape[1]) * np.array( detectCoords[:,0] > 0 ) * np.array( detectCoords[:,1] > 0) ]

    # Return array of coordinates within detectCoords that are 1 w/ respect to global valid coordinates (open coordinates)
    openCoords = detectCoords_pruned[ fly.validCoords[ detectCoords_pruned[:,0], detectCoords_pruned[:,1] ] == 1 ]

    # If exactly one open coordinate was found in detection radius:
    if len(openCoords) == 1:
        # Determine shortest open distance from fly centroid and calculate absolute angle of that point
        openDistances = np.linalg.norm(openCoords - fly.curPos)
        openCoordsMinDist = np.squeeze(openCoords[ openDistances==np.min(openDistances)])
        openAngle = getAngleAbs(fly.curPos, openCoordsMinDist)
        openDist = np.linalg.norm(openCoordsMinDist - fly.curPos)

    # If more than one open coordinate was detected
    elif len(openCoords) > 1:
        # If more than one open coordinate found, iterate over resulting angles and choose point that results in the angle closest to current absolute heading angle
        openAngles = np.zeros(len(openCoords))
        angleDiff = np.zeros(len(openCoords))
        openDistances = np.linalg.norm(openCoords - fly.curPos, axis=1)
        # Iterate through open coordinates and find those with lowest angle difference to current absolute heading direction
        for od in range(0,len(openDistances)):
            openAngles[od] = getAngleAbs(fly.curPos, openCoords[od])
            angleDiff[od] = getAngleDiff(fly.lastAngleAbs, openAngles[od])

        # Assign returned openAngle as the angle with the smallest difference to current heading direction (If true for multiple angles, take the first one)
        minAngleDiffID = np.where(angleDiff==min(angleDiff))[0]
        openAngle = openAngles[ minAngleDiffID ][0]
        # Same for open coordinate distance
        openDist = openDistances[ minAngleDiffID][0]

    # If no open coordinate was in detection range
    else:
        print('ERROR: No available coordinate reachable')
        openAngle = None
        openCoordsMinDist = None
        openDist = None
        return openAngle, openDist


    return openAngle, openDist


# Choose speed based on previously chosen new angle (speed empirically depends on angle)
def chooseSpd(fly, mu=5, av_sigma=1, spdVarInc=0.1):

    # Set just utilized speed as last speed
    fly.lastSpd = fly.curSpd
    # Choose new speed from speed distributions relative to current angle heading
    # Note: Currently sampling from Gaussian
    fly.curSpd = np.random.normal( mu, av_sigma + (fly.OOB * spdVarInc) )

    return fly

# Update new fly position based on chosen angle and speed
def updatePos(fly, wallFollowing=True, wallBias=0.1, detectRadius=1.5):

    # Determine last absolute heading direction
    fly.lastAngleAbsBackUp = fly.lastAngleAbs
    #fly.lastAngleAbs = findatan2( fly.curPos[0] - fly.lastPos[0], fly.curPos[1] - fly.lastPos[1] )
    fly.lastAngleAbs = fly.curAngleAbs
    # If last and current position are the same, last absolute angle is NaN; reassign old heading angle
    if np.isnan(fly.lastAngleAbs):
        print('last Angle should not be 0!')
        fly.lastAngleAbs = fly.lastAngleAbsBackUp

    # If fly should wall follow, adjust mean of heading direction distribution to be mixed between straight ahead & closest wall angle (weighted by wallBias)
    # Note that wall 'attraction' increases as distance to wall decreases
    if wallFollowing:
        wallAngle, wallDist = detectWall(fly, detectRadius=detectRadius)

        # If wall was successfully detected:
        if wallAngle is not None:
            # Take the wallBias weighted wallAngle and last heading angle to update proposed angle before adding relative heading angle later
            fly.lastAngleAbs = getWeightedAngleMean(wallAngle, fly.lastAngleAbs, wallBias)

    # If fly proposed an out of bounds coordinate on the last cycle already, update last absolute heading angle to an open coordinate
    if fly.OOB > 0:
        fly.lastAngleAbs,_ = detectOpenCoords(fly, detectRadius=detectRadius)

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
    if (np.any(flyBodyCoords < 0) or np.any(flyBodyCoords >= np.asarray([YmazeXmax, YmazeYmax]))) or (not all( fly.validCoords[ flyBodyCoords[:,0], flyBodyCoords[:,1] ])):
        # Fly outside of maze array bounds, reset porposed position and return early
        if fly.OOB > 1000:
           print('ERROR: Stuck at position',fly.lastPos, 'with angle:', fly.curAngleAbs, 'with rel angle pull:', fly.curAngleRel, 'into proposed position:', fly.curPos)
        fly.curPos = fly.lastPos.copy()
        fly.lastPos = fly.lastPosBackUp.copy()
        fly.curAngleAbs = fly.lastAngleAbs
        fly.lastAngleAbs = fly.lastAngleAbsBackUp
        # Also report that fly would have been out of bounds
        fly.OOB += 1
        # Return early
        # print('out of bounds at ', fly.curPos)
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
        chooseAngle(fly, av_sigma=av_sigma, brownMotion=brownMotion)
        updatePos(fly, wallFollowing=wallFollowing, wallBias=wallBias, detectRadius=detectRadius)

        # Keep track of overall cycles to read out if fly got 'stuck' and put an upper bound on runtime
        cycle += 1

        # If proposed position was accepted, update turn and log frame data, then advance frame
        if fly.OOB == 0:
            updateTurn(fly, bArmPoly, lArmPoly, rArmPoly)
            expmt[frame,0] = fly.curPos[0].copy()
            expmt[frame,1] = fly.curPos[1].copy()
            expmt[frame,2] = np.squeeze(fly.nTurn)
            expmt[frame,3] = fly.curTurn
            expmt[frame,4] = fly.startArmTurn
            expmt[frame,5] = fly.curAngleAbs
            expmt[frame,6] = fly.curAngleRel
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


def runExperiment(flyN, Ymaze, imgYmaze, bArmPoly, lArmPoly, rArmPoly, data = None, duration=30*60*60, flySpd=5, angleBias=0.5,  av_sigma=10, bodySize=30, visualize=False, cleanup=True, brownMotion=False, wallFollowing=True, wallBias=0.1, detectRadius=1.5):

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
    