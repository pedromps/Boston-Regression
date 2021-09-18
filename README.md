# Boston-Regression
 
The Boston Housing dataset can be used to train regression models to predict the cost of houses. In this repository two methods for regression are implemented: a linear regression and a (deep) neural network.


The neural network is simple and consists only of 3 dense layers with parameters arbitrarily chosen, which produced decent results (with exception of the output layer which natural could only output 1 value for this regression problem). 


The linear regression has the particularity of having as imput a part and not the totality of the training data. The chosen part of the data for this method corresponds to the features that had the highest abolute correlation (that is, values of correlation above 0.5 or below -0.5).


The neural network performed best in the metrics implemented (MSE, RMSE and MAE).