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
    if key == "normalised":
        y_test = (Y-y_min)/(y_max-y_min)
    else:
        y_test = Y
    
    return x_train, x_test, y_train, y_test, [y_max, y_min]

# function to calculate any of these 3 metrics
def metric_calc(y_t, y_p, met): 
    if met=="MSE" or met=="mse":
        metric = np.mean((np.array(y_t)-y_p)**2)
    elif met=="MAE" or met=="mae":
        metric = np.mean(np.abs(np.array(y_t)-y_p))
    elif met=="RMSE" or met=="rmse":
        metric = np.sqrt(np.mean((np.array(y_t)-y_p)**2))
    return metric

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

#-----------------------------------------------------------------------------
metric = "MAE"


# data = corr_data
#the value to predict is in the last column
x_train, x_test, y_train, y_test = train_test_split(boston, boston_t, train_size = 0.8)

#normalise in respect to the training data
mode = "n" #<- "normal" to view normalised metrics, any other input for denormalised.
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
y_pred_nn = model.predict(x_test)

#denormalise to get the real world values
if mode != "normalised":
    y_pred_nn = y_pred_nn*(y_params[0]-y_params[1]) + y_params[1]


#calculate and print metric
res = metric_calc(y_test, y_pred_nn, metric)
print(metric+" of the data = ", res)




#data with only the highly correlated dat, for the linear regression
x_train_lin = x_train[aux.index]
x_test_lin = x_test[aux.index]

#linear regression
lin_model = LinearRegression()
lin_model.fit(x_train_lin, y_train)

y_pred_lin = lin_model.predict(x_test_lin)
if mode != "normalised":
    y_pred_lin = y_pred_lin*(y_params[0]-y_params[1]) + y_params[1]
    
res = metric_calc(y_test, y_pred_lin, metric)
print(metric+" of the data = ", res)
