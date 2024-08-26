# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:15:17 2024

@author: furko
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Loading DataSet
data = pd.read_csv("Weather_Data.csv")
data = data.drop(["Date"], axis=1)

# independent variables are converted to binary format
data["WindGustDir"] = pd.factorize(data["WindGustDir"])[0]
data["WindDir9am"] = pd.factorize(data["WindDir9am"])[0]
data["WindDir3pm"] = pd.factorize(data["WindDir3pm"])[0]
data["RainToday"] = pd.factorize(data["RainToday"])[0]
data["RainTomorrow"] = pd.factorize(data["RainTomorrow"])[0]

# Dependent and independent variables has been setted
Y = data.iloc[:,-1]
X = data.drop(["RainTomorrow"], axis = 1)

scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)

# train and test splitting
train_x, test_x, train_y, test_y = train_test_split(scaled_data, Y, test_size=0.2, random_state=1)
#----------------------------------------------------------------------------------------------------
# Linear Model processing
linearModel = LinearRegression()
linearModel.fit(train_x, train_y)
print("\nLinear regression stats:")

print("Mean Absolute Error:", mean_absolute_error(test_y, linearModel.predict(test_x)))

print("Mean Squared Error:", mean_squared_error(test_y, linearModel.predict(test_x)))

print("R2-Score:", r2_score(test_y, linearModel.predict(test_x)))

#----------------------------------------------------------------------------------------------------
logisticModel = LogisticRegression(solver="liblinear", C=0.01)
logisticModel.fit(train_x, train_y)
print("\nLogistic Regression stats:")

print("Accuracy Score:", accuracy_score(test_y, logisticModel.predict(test_x)))

print("Jaccard Index:", jaccard_score(test_y, logisticModel.predict(test_x)))

print("F1-Score:", f1_score(test_y, logisticModel.predict(test_x)))

print("LogLoss:", log_loss(test_y, logisticModel.predict_proba(test_x)[:, 1]))
#----------------------------------------------------------------------------------------------------
treeModel = DecisionTreeClassifier(criterion="entropy", max_depth=4)
treeModel.fit(train_x, train_y)
print("\nDecission Tree stats: ")

print("Accuracy Score:", accuracy_score(test_y, treeModel.predict(test_x)))

print("Jaccard Index:", jaccard_score(test_y, treeModel.predict(test_x)))

print("F1-Score:", f1_score(test_y, treeModel.predict(test_x)))

print("LogLoss:", log_loss(test_y, treeModel.predict_proba(test_x)[:, 1]))

#----------------------------------------------------------------------------------------------------
KNNModel = KNeighborsClassifier(n_neighbors=25)
KNNModel.fit(train_x, train_y)
print("\nKNN stats:")

print("Accuracy Score:", accuracy_score(test_y, KNNModel.predict(test_x)))

print("Jaccard Index:", jaccard_score(test_y, KNNModel.predict(test_x)))

print("F1-Score:", f1_score(test_y, KNNModel.predict(test_x)))

print("LogLoss:", log_loss(test_y, KNNModel.predict_proba(test_x)[:, 1]))

#----------------------------------------------------------------------------------------------------
SVModel = svm.SVC(kernel="rbf", C=0.01,  probability=True)
SVModel.fit(train_x, train_y)
print("\nSVM stats:")

print("Accuracy Score:", accuracy_score(test_y, SVModel.predict(test_x)))

print("Jaccard Index:", jaccard_score(test_y, SVModel.predict(test_x)))

print("F1-Score:", f1_score(test_y, SVModel.predict(test_x)))

print("LogLoss:", log_loss(test_y, SVModel.predict_proba(test_x)[:, 1]))