# -*- coding: utf-8 -*-
"""
Created on Wed May 23 21:41:10 2018

@author: Grenceng
"""
import numpy as np

def peakDetection(mX, t):
	thresh = np.where(np.greater(mX[1:-1],t), mX[1:-1], 0); # locations above threshold
	next_minor = np.where(mX[1:-1]>mX[2:], mX[1:-1], 0)     # locations higher than the next one
	prev_minor = np.where(mX[1:-1]>mX[:-2], mX[1:-1], 0)    # locations higher than the previous one
	ploc = thresh * next_minor * prev_minor                 # locations fulfilling the three criteria
	ploc = ploc.nonzero()[0] + 1                            # add 1 to compensate for previous steps
	return ploc

def localMinMax(mX):
    local_max = np.where(mX[1:-1]>mX[:-2], mX[1:-1], 0)    # locations higher than the previous one
    local_min = np.where(mX[1:-1]<mX[:-2], mX[1:-1], 0)    # locations lower than the previous one
    local_max = local_max.nonzero()[0] + 1                            # add 1 to compensate for previous steps
    local_min = local_min.nonzero()[0] + 1                            # add 1 to compensate for previous steps
    return local_max, local_min