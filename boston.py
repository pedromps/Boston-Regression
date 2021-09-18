# -*- coding: utf-8 -*-
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
# from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

boston_dataset = load_boston()
print(boston_dataset.keys())

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)


# lin_model = LinearRegression()
# lin_model.fit(X_train, Y_train)

data = np.array(boston)
#the value to predict is in the last column
x_train, x_test, y_train, y_test = train_test_split(data[:, 0:data.shape[1]-1], data[:, -1], train_size = 0.8)

#normalise in respect to the training data
x_max = np.nanmax(x_train, axis=0)
x_min = np.nanmin(x_train, axis=0)
y_max = np.nanmax(y_train)
y_min = np.nanmin(y_train)

x_train = (x_train-x_min)/(x_max-x_min)
x_test = (x_test-x_min)/(x_max-x_min)

y_train = (y_train-y_min)/(y_max-y_min)
# y_test =  (y_test-y_min)/(y_max-y_min)


#the model
model = Sequential()
model.add(Dense(64, input_dim = 12 , activation = 'tanh'))
model.add(Dense(6, activation = 'relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.summary()
history = model.fit(x_train, y_train, batch_size = 6, epochs=100)

#predicting 
y_pred = model.predict(x_test)

#denormalise to get the real world values
y_pred = y_pred*(y_max-y_min) + y_min

#calculate MSE
MSE = np.mean((y_test.ravel()-y_pred.ravel())**2)
print("MSE of the normalised data = ", MSE)
