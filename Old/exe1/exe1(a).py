# 2022-11-28 10:48:59
import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics as mt
import pandas as pd
#Importing dataset
data = pd.read_csv("resources/winequality-white.csv", sep=";")
X = data.iloc[:,:-1]
Y= data.iloc[:,-1]
Y = pd.DataFrame(Y)
#Dividing into test and training sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)

#Creating model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train,Y_train)
#Predicting result
Y_pred = model.predict(X_test)