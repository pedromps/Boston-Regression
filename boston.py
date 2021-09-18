# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import ModelCheckpoint


boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston_t = pd.DataFrame(boston_dataset.target)

# lin_model = LinearRegression()
# lin_model.fit(X_train, Y_train)

data = np.array(boston)
target = np.array(boston_t)
#the value to predict is in the last column
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size = 0.8)

#normalise in respect to the training data
x_max = np.nanmax(x_train, axis=0)
x_min = np.nanmin(x_train, axis=0)
y_max = np.nanmax(y_train)
y_min = np.nanmin(y_train)

x_train = (x_train-x_min)/(x_max-x_min)
x_test = (x_test-x_min)/(x_max-x_min)

y_train = (y_train-y_min)/(y_max-y_min)
# y_test =  (y_test-y_min)/(y_max-y_min)


#the callback and model
# checkpoint_filepath = "./tmp"
# model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)

model = Sequential()
model.add(Dense(64, input_dim = 13 , activation = 'relu'))
model.add(Dense(6, activation = 'relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.summary()
history = model.fit(x_train, y_train, batch_size = 32, epochs=200, validation_split=0.125)

#plotting
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()

#predicting 
y_pred = model.predict(x_test)

#denormalise to get the real world values
y_pred = y_pred*(y_max-y_min) + y_min

#calculate MAE or MSE
#MSE = np.mean((y_test.ravel()-y_pred.ravel())**2)
MAE = np.mean(np.abs(y_test.ravel()-y_pred.ravel()))
print("MAE of the normalised data = ", MAE)
