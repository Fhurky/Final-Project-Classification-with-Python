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
test_x, train_x, test_y, train_y = train_test_split(scaled_data, Y, test_size=0.2, random_state=1)

# Model training
linearModel = LinearRegression()
linearModel.fit(train_x, train_y)

logisticModel = LogisticRegression(solver="liblinear", C=0.01)
logisticModel.fit(train_x, train_y)

treeModel = DecisionTreeClassifier(criterion="entropy", max_depth=4)
treeModel.fit(train_x, train_y)

KNNModel = KNeighborsClassifier(n_neighbors=5)
KNNModel.fit(train_x, train_y)

SVModel = svm.SVC(kernel="rbf", C=0.01)
SVModel.fit(train_x, train_y)