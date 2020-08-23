# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 12:55:21 2020

This is a sample Neural Network strategy adapted from a quantitative finance blog. 

@author: Amogh
"""

import numpy as np
import pandas as pd
import talib as ta
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import random
random.seed(10)


# Importing and preparing the data:
start = dt.datetime.today()-dt.timedelta(5900)
end = dt.datetime.today()
df = yf.download("MARUTI.NS", start, end)
df = df.dropna()
df = df[['Open', 'High', 'Low', 'Close']]

# Input features:
df['H-L'] = df['High'] - df['Low']
df['C-O'] = df['Close'] - df['Open']
df['50day MA'] = df['Close'].shift(1).rolling(50).mean()
df['10day MA'] = df['Close'].shift(1).rolling(10).mean()
df['30day MA'] = df['Close'].shift(1).rolling(30).mean()
df['Std_dev'] = df['Close'].rolling(5).std()
df['RSI'] = ta.RSI(df['Close'].values, timeperiod=14)
df['Williams %R'] = ta.WILLR(df['High'].values, df['Low'].values, df['Close'].values, 7)

# Defining the OUTPUT VALUE AS PRICE RISE : (1 if Price rises the next day)
df['Price_Rise'] = np.where(df['Close'].shift(-1)>df['Close'],1,0)
df = df.dropna()

# Creating Dataframes X and y for storing Input and Output variables:
X = df.iloc[:, 4:-1]
y = df.iloc[:, -1]

# Spliting the dataset:
split = int(0.8*len(df))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Feature Scaling:
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# xxxxxxxxxxxxxxxx    BUILDING THE ARTIFICIAL NEURAL NETWORK     xxxxxxxxxxxxxxxxxxx

clf = Sequential()
clf.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1])) #first layer
clf.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu')) # second and last hidden layer
clf.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))   # output layer
# Dense function: Units = defines no.of nodes/neurons
#                 Kernel_initializer = dfines starting values for weights. We've set 'uniform- - values from unifrom distribution
#                 Activation = activation of neurons in this hidden layer. 'Relu'=Rectified Linear Unit
#                 Input_dim = Defines no.of inputs, Here it is equal to no.of columns of our input feature 
clf.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy']) # Compiling the classifier
# Compile function: Optimiizer = 'adam' - extension of stochastic gradient descent
#                   Loss = defines the loss to be optimized during training period
#                   Metrics = list of metrics to be evaluated. Here, Accuracy is our evaluation metric 

# FITTING THE NEURAL NETWORK:
clf.fit(X_train, y_train, batch_size = 10, epochs = 100)
# batch_size: No.of data points used to compute error before Backpropagating the errors and modifying the weights.
# epochs = No.of times the training of the model will be performed on training set.


#          XXXXXXXXXX     PREDICTING THE STOCK MOVEMENT     XXXXXXXXXX
y_pred = clf.predict(X_test)
y_pred = (y_pred > 0.5)  # converting y_pred to binary values - True or False depending on >0.5 or <0.5

# Creating y_pred column in df:
df['y_pred'] = np.NaN
df.iloc[(len(df)-len(y_pred)):,-1:] = y_pred  #storing values of y_pred
trade_df = df.dropna()                        #dropping NaN values and storing them in Trade_df

# Computing Strategy Returns:
trade_df['Tomorrows Returns'] = 0.
trade_df['Tomorrows Returns'] = np.log(trade_df['Close']/trade_df['Close'].shift(1))
trade_df['Tomorrows Returns'] = trade_df['Tomorrows Returns'].shift(-1)

trade_df['Strategy Returns'] = 0.
trade_df['Strategy Returns'] = np.where(trade_df['y_pred'] == True, trade_df['Tomorrows Returns'], -trade_df['Tomorrows Returns'])

# Cumulative returns for both; market and strategy:
trade_df['Cumulative Market Returns'] = np.cumsum(trade_df['Tomorrows Returns'])
trade_df['Cumulative Strategy Returns'] = np.cumsum(trade_df['Strategy Returns'])


# Visualizing the Returns:
plt.figure(figsize = (8, 4))
plt.plot(trade_df['Cumulative Market Returns'], color='r', label='Market Returns')
plt.plot(trade_df['Cumulative Strategy Returns'], color='g', label='Strategy Returns')
plt.legend()
plt.show()







