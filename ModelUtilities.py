'''
This file contains a bunch of model utilities for training and testing model
Author:     Henry Dikeman
Email:      dikem003@umn.edu
Date:       03/26/21
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# configure plotting params
plt.style.use('seaborn-bright')

# trains a neural net using a dataloader and the models loss/optimizer functions
def train_model(model, train_loader, loss_fn, optimizer, device):
    # set model to training mode
    model.train()
    # set loss to 0 at each epoch
    loss = 0
    # loop through all training data in dataloader
    for batch_features, batch_results in tqdm(train_loader, ncols=100):
        # send batch of data to device
        batch_features = batch_features.to(device)
        batch_results = batch_results.to(device)

        # reset gradient back to 0
        optimizer.zero_grad()

        # compute feedforward estimation of data in AE
        predictions = model(batch_features)

        # calculate training loss for batch
        batch_loss = loss_fn(predictions, batch_results.unsqueeze(1))

        # calculate batch gradient
        batch_loss.backward()

        # update parameters by batch gradient
        optimizer.step()
        
        # add batch loss to epoch loss
        loss += batch_loss

    # calculate average MSE for epoch
    loss /= len(train_loader)

    # return loss to exterior function
    return loss

# tests model using validation dataloader and model loss function
def test_model(model, test_loader, loss_fn, device):
    # set to eval mode, get size and initialize test loss
    model.eval()
    test_accuracy = 0

    # calculate testing data accuracy
    with torch.no_grad():
        for batch_features, batch_results in test_loader:
            # send batch of data to device
            batch_features = batch_features.to(device)
            batch_results = batch_results.to(device)

            # predict batch features of validation set
            predictions = torch.round(model(batch_features))
            
            # calculate correct predictions
            test_accuracy += (predictions == batch_results.unsqueeze(1)).sum().float()
    
    # calculate average loss
    print('raw correct:',test_accuracy)
    print('test loader size:',len(test_loader))
    test_accuracy /= 1200000
    test_accuracy *= 100

    return test_accuracy

def plot_training(train_loss, test_loss, model_name):
    # make x-axis values for numpy of epochs
    epochs = np.arange(train_loss.shape[0])
    
    # plot training progress graph
    plt.plot(epochs, train_loss, 'k-', label='Training Loss')
    plt.plot(epochs, test_loss, 'r-', label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Progress for '+model_name)
    plt.legend()
    plt.show()
