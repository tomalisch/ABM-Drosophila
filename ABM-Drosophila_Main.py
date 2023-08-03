### Agent-based model of Drosophila decision-making

## Dependencies
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

## Set up Y-maze map as binary array; size equals empirical data; each cell equals a pixel
print('Initializing Y-maze map...')

Ymaze = np.zeros([8,8], dtype=bool)
sns.heatmap(Ymaze)

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
        startPos = tuple( validPositions[ random.randint(0,len(validPositions)-1) ] )
    fly = flyAgent()
    fly.curPos = startPos
    return fly