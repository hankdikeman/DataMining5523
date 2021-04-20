"""
This file is a quick visualization of the different csv label data for the 5 1000x1000 images
Author: Henry Dikeman
Email:  dikem003@umn.edu
Date:   04/20/21
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from ImageIOFunctions import LoadImageFromCSV

# tests for visualization of label data
if __name__ == "__main__":
    # store CL argument to determine which image to show
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', action='store', dest='filename', help='Filename of file to be visualized')
    args = parser.parse_args()

    # generate filename
    fname = os.path.join(os.getcwd(),args.filename)
    # load data to numpy array with helper function
    labels = LoadImageFromCSV(fname)

    # display labeled image with imshow
    plt.imshow(labels, cmap='gray', vmin=0, vmax=1)
    plt.show()

