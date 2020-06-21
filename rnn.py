"""

This RNN is used for predicting stock trends of the Google stock.
@Editor: Eric Albrecht

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Part 1 - Data Preprocessing

# Training set is only used, not test set.
# Once training is done, then the test set will be introduced.

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

# This creates a numpy array rather than a simple vector
# This grabs the open column and makes it into a numpy array
training_set = dataset_train.iloc[:,1:2].values

#This works because the stock prices will be normalized between 0 & 1
sc = MinMaxScaler(feature_range = (0,1), copy=True)

#This method fits the data (finds min & max) to apply normalization
# The transform methods computes the standardized 
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 time steps and 1 output
# the 60 time steps is a tested value, that has to be iterated over the model to find.
# This means that each day looks at the three previous months to predict the price
x_train = []
y_train = []

# the 60 refers to the three months, 1258 is total # of prices
for i in range(60, 1258):
    # This gets the range of all of the 60 last stock prices
    x_train.append(training_set_scaled[i-60:i,0])
    #uses the next stock price as the output prediction
    y_train.append(training_set_scaled[i,0])

# Part 2 - Building the RNN



# Part 3 - Making the predictions and visualizing the results

# Number of time steps - learning what the data structure for remembering should be
