# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import ModelCheckpoint
# import seaborn as sns 

# function to normalise in respect to training data. This means that the data
# is normalised with the minima and maxima of the training set. That is, both
# the training and testing data are normalised with the values calculated from
# the training data.
def normalise_to_train(x, X, y, Y, key):
    x_max = np.nanmax(x, axis=0)
    x_min = np.nanmin(x, axis=0)
    y_max = np.nanmax(y)
    y_min = np.nanmin(y)
    
    x_train = (x-x_min)/(x_max-x_min)
    x_test = (X-x_min)/(x_max-x_min)
    
    y_train = (y-y_min)/(y_max-y_min)
    
    # particularity: if the outputs are denormalised to calculate the metrics
    if key == "normal":
        y_test = (Y-y_min)/(y_max-y_min)
    else:
        y_test = Y
    
    return x_train, x_test, y_train, y_test, [y_max, y_min]

boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston_t = pd.DataFrame(boston_dataset.target)

#to give a name to the target
boston_t.rename(columns={0: 'Price'}, inplace=True)


#to choose only the features with high correlation, the first step is to see how correlated they are
full_data = pd.concat((boston, boston_t), axis=1)
correlation_matrix = full_data.corr().round(2)
# annot = True to print the values inside the square
# uncomment the following line to visualize correlation:
# sns.heatmap(data=correlation_matrix, annot=True)

#high correlation -> >=0.5. correlation with self = 1 so it's excluded posteriously
aux = correlation_matrix.loc[np.abs(correlation_matrix.loc['Price'])>=0.5, 'Price']
aux.drop(index='Price', inplace=True)

#data with only the highly correlated dat
corr_data = boston[aux.index]


#full data
data = np.array(boston)
target = np.array(boston_t)




# data = corr_data
#the value to predict is in the last column
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size = 0.8)


#normalise in respect to the training data
mode = "normal" #<- "normal" to view normalised metrics, any other input for denormalised.
x_train, x_test, y_train, y_test, y_params = normalise_to_train(x_train, x_test, y_train, y_test, mode)


#the callback and model
# checkpoint_filepath = "./tmp"
# model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)

model = Sequential()
model.add(Dense(64, input_dim = x_train.shape[1] , activation = 'relu'))
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
y_pred = y_pred*(y_params[0]-y_params[1]) + y_params[1]

#calculate MAE or MSE
#MSE = np.mean((y_test.ravel()-y_pred.ravel())**2)
MAE = np.mean(np.abs(y_test.ravel()-y_pred.ravel()))
print("MAE of the normalised data = ", MAE)


#linear regression
lin_model = LinearRegression()
lin_model.fit(x_train, y_train)

y_predi = lin_model.predict(x_test)
y_predi = y_predi*(y_params[0]-y_params[1]) + y_params[1]
RMSE = np.mean(np.sqrt((y_test.ravel()-y_predi.ravel())**2))
print("RMSE of the normalised data = ", RMSE)
