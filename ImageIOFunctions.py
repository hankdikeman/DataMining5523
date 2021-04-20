"""
This file contains some simple IO functions to read and write image data from CSV
Author: Henry Dikeman
Email:  dikem003@umn.edu
Date:   04/20/21
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# helper function to load label data from CSV and trim header and label column
def LoadImageFromCSV(fname):
    # load filename using numpy builtin
    rawlabels = np.genfromtxt(fname, dtype=np.float32, delimiter=',')
    # trim header and column label
    labels = rawlabels[1:,1:]
    # return trimmed label data array
    return labels
