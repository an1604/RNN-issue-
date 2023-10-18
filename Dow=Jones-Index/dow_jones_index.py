# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 08:28:08 2023

@author: adina
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# improting the dataset and making training dataset 
dataset =pd.read_csv('dow_jones_index.data')

# getting the training data 
dataset_train =  pd.DataFrame(dataset, columns=['open','close' ,'percent_change_price','percent_change_volume_over_last_wk', 'days_to_next_dividend' ,'percent_return_next_dividend' , 'percent_change_next_weeks_price'])

# manipulate the data 
# convert the 'open' and 'close' prices to float
dataset_train['open'] = dataset_train['open'].str.replace('$', '').apply(float)
dataset_train['close'] = dataset_train['close'].str.replace('$', '').apply(float)

 # feature scaling 
from sklearn.preprocessing import MinMaxScaler
sc =MinMaxScaler(feature_range=(0,1))
dataset_train = sc.fit_transform(dataset_train)

# getting X_train and y_train
X_train = [] 
y_train = [] 
for i in range(11 ,750 ):
    X_train.append(dataset_train[i-11:i])
    y_train.append(dataset_train[i][6])

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = np.reshape(X_train,(X_train.shape[0] ,X_train.shape[1] , 7))

# Buliding the model 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.regularizers import l2

regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 7), kernel_regularizer=l2(0.01)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, kernel_regularizer=l2(0.01)))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, kernel_regularizer=l2(0.01)))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, kernel_regularizer=l2(0.01)))
regressor.add(Dropout(0.2))


# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
