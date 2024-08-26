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

data = pd.read_csv("Weather_Data.csv")
data = data.drop(["Date"], axis=1)

data["WindGustDir"] = pd.factorize(data["WindGustDir"])[0]
data["WindDir9am"] = pd.factorize(data["WindDir9am"])[0]
data["WindDir3pm"] = pd.factorize(data["WindDir3pm"])[0]
data["RainToday"] = pd.factorize(data["RainToday"])[0]
data["RainTomorrow"] = pd.factorize(data["RainTomorrow"])[0]