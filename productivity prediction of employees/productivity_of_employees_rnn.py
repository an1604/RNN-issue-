# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:20:31 2023

@author: adina
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
dataset = pd.read_csv('garments_worker_productivity.csv')
new_ds = dataset
columns_to_remove = ['date', 'quarter', 'day']
new_ds = new_ds.drop(columns_to_remove, axis=1)


# encoding the department column
le = LabelEncoder()
new_ds['department'] =le.fit_transform(new_ds['department'])

# splitting the dataset 
training_set = new_ds[:960]
test_set = new_ds[960:]

# split the data
X = training_set.iloc[: , :-1].values
y = training_set.iloc[: , -1].values

# feature scaling 
sc =MinMaxScaler(feature_range=(0,1))
X=sc.fit_transform(X)

# creating a data structure with 60 timesteps and 1 output
X_train=[]
y_train=[]

for i in range(60 ,960):
    X_train.append(X[i-60:i])
    y_train.append(y[i])
    
X_train, y_train = np.array(X_train), np.array(y_train)

# reshaping 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 11))

#Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialising the RNN
regressor = Sequential()

#Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 25, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.1))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 25, return_sequences = True))
regressor.add(Dropout(0.1))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 25, return_sequences = True))
regressor.add(Dropout(0.1))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 25))
regressor.add(Dropout(0.1))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 35, batch_size = 16)

