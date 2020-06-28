"""

This RNN is used for predicting stock trends of the Google stock.
@Editor: Eric Albrecht

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


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
    
# This converts the data to user arrays for tensorflow
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping, only done to inputs to add dimensions
# Use the keras library for this
# input dimensions can refer to the same stock stats, or comparing stocks
# the shape function gets the size of the axis specified, can use more than 2 dimensions
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Part 2 - Building the RNN

# This is a stacked LSTM with dropout regularization
# This is called regressor due to its continuous output
# Regression is for predicting continuous, classification is for predicting finite output
regressor = Sequential()

# Adding the first LSTM Layer and some dropout regularization
# Units -> The number of LSTM cells or modules in the layer
# Return Sequences -> True, because it will have several LSTM layers. False when you're on the last layer
# Input Shape -> Shape of the input containing x_train
# High dimensionality and lots of neurons in each will help the accuracy
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))

# second step of first LSTM Layer is to add dropout regularization
# Classic # is 20%, aka 20% of neurons will be ignored in forward and backware propagation
regressor.add(Dropout(rate=0.20))

# Add a second LSTM layer with dropout regularization
# because this is the second layer, input layer is not required
# 50 neurons in previous layer is assumed
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.20))

# Add a third LSTM layer with dropout regularization
# Same as second LSTM layer -> both are middle, hidden layers
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.20))

# Add a fourth LSTM Layer with dropout regularization
# 50 units stays the same because this is not the final layer.
# Output layer to follow for the one continuous output of the regression
# Return sequences should be false because no more LSTM modules present
regressor.add(LSTM(units=50))
regressor.add(Dropout(rate=0.20))

# Add a classic fully connected, output layer
# 1 output unit corresponds to the 
regressor.add(Dense(units=1))

# Compile the RNN
# RMS prop is recommended for RNN's but adam used here
# Loss function is mean squared error
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fit the RNN
# X & Y are the input numpy arrays and output numpy arrays
# Batch_size
regressor.fit(x=x_train, y=y_train, batch_size=32, epochs=100)

# Part 3 - Making the predictions and visualizing the results

# Avoid overfitting of the training set because then it won't have enough variance to recognize other test sets

# Getting the real stock price open of 2017
actual_results = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = actual_results.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
# Need the 60 previous days to create a concatenated data
dataset_total = pd.concat((dataset_train['Open'], actual_results['Open']), axis=0)
inputs = dataset_total[len(dataset_total)-len(actual_results)-60:].values
inputs = inputs.reshape(-1, 1)
# The object doesn't need to be fit, only scaled
inputs = sc.transform(inputs)
# only scale the inputs, not the test values
# Visualize the results

x_test = []

# the 60 refers to the three months, 80 refers to the predicted month
for i in range(60, 80):
    # This gets the range of all of the 60 last stock prices
    x_test.append(inputs[i-60:i,0])
    
# This converts the data to user arrays for tensorflow
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# This is the prediction part of the RNN
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualization of prices using matplotlib
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time (Days)')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.show()