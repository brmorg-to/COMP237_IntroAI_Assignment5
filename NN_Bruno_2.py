#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 14:22:10 2021

@author: brunomorgado
"""

import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl



'''Exercise # 2: Multi-layer feed forward to recognize sum pattern'''

#Set the seed and generate two arrays of random numbers
np.random.seed(1)
set1 = np.random.uniform(-0.6, 0.6, 10).reshape(10,1)
set2 = np.random.uniform(-0.6, 0.6, 10).reshape(10,1)

#Concatenate the two samples
input_bruno = np.concatenate((set1, set2), axis=1)

#Inspect the concatenated array
print(input_bruno)
print(input_bruno.shape)

#Calculate the target variable based on the two inputs
target = (input_bruno[:, 0] + input_bruno[:, 1]).reshape(10,1)

print(target)

# Minimum and maximum values for each dimension
dim1_min, dim1_max = input_bruno[:,0].min(), input_bruno[:,0].max()
dim2_min, dim2_max = input_bruno[:,1].min(), input_bruno[:,1].max()


# Define a single-layer neural network with 2 hidden layers. One with 5 and the other with 3 neurons
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]

nn = nl.net.newff([dim1, dim2], [5,3,1])

# Set the training algorithm to gradient descent
nn.trainf = nl.train.train_gd

# Train the neural network
error = nn.train(input_bruno, target, epochs=1000, show=100, goal=0.00001)


print(f"Number of inputs: {nn.ci}")
print(f"Number of outputs: {nn.co}")
print(f"Number of layers, including hidden layers: {len(nn.layers)}")


plt.plot(error)
plt.xlabel('Epoch number')
plt.ylabel('Train error')
plt.grid()
plt.show()


#Predict a value with one single test sample
result2 = nn.sim([[0.1,0.2]])

print(f'Result 2: {result2}')
