"""
This neural net is designed to perform classification of farmland areas from pixel values
Author:     Henry Dikeman
Email:      dikem003@umn.edu
Date:       04/02/21
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
import argparse
import os
import sys
from ModelUtilities import train_model, test_model, plot_training
from ImageIOFunctions import LoadImageFromCSV

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## train data
class trainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index,:], self.y_data[index]
        
    def __len__ (self):
        return self.X_data.shape[0]

## train data
class testData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index,:], self.y_data[index]
        
    def __len__ (self):
        return self.X_data.shape[0]
    
# General neural net class
class ResidDerivEstimator(nn.Module):
    """
    This class implements a decoder or encoder for the autoencoder class. Future changes would likely involve changing from this bootleg format to a nn.Sequential structure for simplicity (with '*' unpack operation for variable layer quantities)
    """
    # initialize model
    def __init__(self, n_vars, n_output, n_blocks, size_internal):
        super(ResidDerivEstimator, self).__init__()

        # dense input layer
        self.input_layer = nn.Linear(n_vars, size_internal)

        # number of hidden layer for looping operations
        self.n_resid_block = n_blocks

        # loop to generate uniform dense hidden layers and batchnorm layers
        for i in range(n_blocks):
            # hidden layer
            setattr(self, "r"+str(i), ResidualBlock(size_internal, size_internal*2))

        self.hidden_layer = nn.Sequential(
            nn.Linear(size_internal, size_internal*2),
            nn.ReLU(), # nn.Hardtanh(min_val=-5, max_val=5),
            nn.Dropout(0.01)
        )

        # output layer with specified shape
        self.output_layer = nn.Sequential(
            nn.Linear(size_internal*2, n_output),
            nn.Sigmoid()
        )

    # feedforward calculation
    def forward(self, x):
        # pass through input layer
        x = self.input_layer(x)

        # store residual of x
        residual = x

        # loop through nested hidden layer + LR activation
        for i in range(self.n_resid_block):
            x = getattr(self,"r"+str(i))(x)

        # add in residual
        x = x + residual

        # final hidden layer
        x = self.hidden_layer(x)

        # pass through output function
        x = self.output_layer(x)

        return x

    # save generated model
    def save(self):
        torch.save(self.state_dict(), "Models/ResidDerivEst.pkl")

    # load existing model from pickle file
    def load(self):
        self.load_state_dict(torch.load("Models/ResidDerivEst.pkl"))

class ResidualBlock(nn.Module):
    '''
    This class implements a residual block to be used for the residual derivative estimator net
    '''
    def __init__(self, layer_size, internal_size):
        super(ResidualBlock, self).__init__()

        # generate batch norm layer
        self.bn = nn.BatchNorm1d(num_features=layer_size, affine=True)

        # final activation layer
        self.rl = nn.ReLU()

        # generate residual block
        self.block = nn.Sequential(
            nn.Linear(layer_size, internal_size),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(internal_size, layer_size),
            nn.LeakyReLU() # Hardtanh(min_val=-5.0, max_val=5.0),
            )

    def forward(self, x):
        # pass input through batch norm layer
        x = self.bn(x)

        # store residual
        residual = x

        # pass input through residual block
        x = self.block(x)

        # incorporate residual
        x = x + residual

        # apply final activation
        x = self.rl(x)

        return x

if __name__ == "__main__":
    # generate argument parser and define args
    prsr = argparse.ArgumentParser()
    prsr.add_argument('-d','--loadderiv',action='store_true',help="boolean flag to load saved derivative estimator")
    
    # parse command line arguments
    args = prsr.parse_args()
    # loading data
    filenames = ['1000_1000.csv','1000_2000.csv','1000_3000.csv','1000_4000.csv','1000_5000.csv','1000_6000.csv']
    # image data red retrieval from csv
    reddatafnames = [os.path.join(os.getcwd(),'data','red'+x) for x in filenames]
    reddatalist = [LoadImageFromCSV(fname) for fname in reddatafnames]
    
    # stack data vertically to produce raw image data array
    redimage_structured = np.vstack(tuple(reddatalist))
    raw_imagered = redimage_structured.flatten()
    # image data blue retrieval from csv
    bluedatafnames = [os.path.join(os.getcwd(),'data','blue'+x) for x in filenames]
    bluedatalist = [LoadImageFromCSV(fname) for fname in bluedatafnames]
    
    # stack data vertically to produce raw image data array
    blueimage_structured = np.vstack(tuple(bluedatalist))
    raw_imageblue = blueimage_structured.flatten()
    # image data green retrieval from csv
    greendatafnames = [os.path.join(os.getcwd(),'data','green'+x) for x in filenames]
    greendatalist = [LoadImageFromCSV(fname) for fname in greendatafnames]
    
    # stack data vertically to produce raw image data array
    greenimage_structured = np.vstack(tuple(greendatalist))
    raw_imagegreen = greenimage_structured.flatten()
    # image data blue retrieval from csv
    nirdatafnames = [os.path.join(os.getcwd(),'data','nir'+x) for x in filenames]
    nirdatalist = [LoadImageFromCSV(fname) for fname in nirdatafnames]
    
    # stack data vertically to produce raw image data array
    nirimage_structured = np.vstack(tuple(nirdatalist))
    raw_imagenir = nirimage_structured.flatten()
    # label data retrieval from csv
    labeldatafnames = [os.path.join(os.getcwd(),'data','label'+x) for x in filenames]
    labeldatalist = [LoadImageFromCSV(fname) for fname in labeldatafnames]
    
    # stack data vertically to produce raw label data array
    labels_structured = np.vstack(tuple(labeldatalist))
    raw_labels = labels_structured.flatten()
    # generate NDVI values
    ndviimage_structured = (nirimage_structured - redimage_structured) / (nirimage_structured + redimage_structured + 0.00001)
    raw_imagendvi = ndviimage_structured.flatten()
    # import train test split function
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    # collapse and collect input channels and store as record array
    raw_image = np.vstack((raw_imagered,raw_imagegreen,raw_imageblue,raw_imagenir,raw_imagendvi)).T
    raw_image = StandardScaler().fit_transform(raw_image)
    # perform train test split
    image_train, image_test, label_train, label_test = train_test_split(raw_image, raw_labels, test_size=0.2)

    print('train size:',image_train.shape)
    print('test size:',image_test.shape)

    # store feature number constant
    NUM_OUTPUT = 1
    NUM_REDUCED = 5
    NUM_HIDDEN = 1
    SIZE_HIDDEN = 8

    train_data = trainData(torch.FloatTensor(image_train),
                           torch.FloatTensor(label_train))
    test_data = testData(torch.FloatTensor(image_test), 
                            torch.FloatTensor(label_test))
    BATCH_SIZE = 2048
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=4096)

    # generate new derivative estimator
    deriv = ResidDerivEstimator(NUM_REDUCED, NUM_OUTPUT, NUM_HIDDEN, SIZE_HIDDEN)

    # generate optimizer and loss calculation
    DE_optim = optim.Adam(deriv.parameters(), lr=1E-3)
    DE_loss_fn = nn.MSELoss()
    
    # print model summary
    print('\nDERIVATIVE ESTIMATOR')
    print(deriv)
    # summary(deriv, (1,NUM_REDUCED), 1024)
    deriv.to(DEVICE)

    if args.loadderiv:
        try:
            print('\n### LOADING MODEL ###\n')
            deriv.load()
            print('### MODEL LOADED ###\n')
        except Exception:
            print("Unexpected error, your model likely isn't saved:", sys.exc_info()[0])
            print("Program will now exit")
            raise
    else:
        # set number of training epochs
        EPOCHS = 5
    
        # make callback arrays
        trainloss_CB = np.zeros((EPOCHS))
        testloss_CB = np.zeros((EPOCHS))
        
        # loop through number of epochs
        for epoch in range(EPOCHS):
            # train model for one epoch
            loss = train_model(deriv, train_loader, DE_loss_fn, DE_optim, DEVICE)
    
            # print MSE batch summary
            print(f"epoch: {epoch+1}/{EPOCHS}, Avg Loss = {loss}")
            trainloss_CB[epoch] = loss
            
            # test model on validation data
            test_loss = test_model(deriv, test_loader, DE_loss_fn, DEVICE)
        
            # print test summary
            print(f"\tTesting Loss: {epoch+1}/{EPOCHS}, Avg Loss = {test_loss}\n")
            testloss_CB[epoch] = test_loss
    
        # print training progress graph
        plot_training(trainloss_CB, testloss_CB, 'Derivative Estimator')
        
        # save model after training
        deriv.save()
    
    # estimate class labels
    deriv.eval()
    with torch.no_grad():
        imagetesttensor = torch.FloatTensor(image_test).to(DEVICE)
        predictedlabels = deriv(imagetesttensor).detach().numpy()
        correct_results_sum = (predictedlabels == label_test).sum().float()
        acc = correct_results_sum/label_test.shape[0]*100

    print('Accuracy on Validation Data:',acc)
