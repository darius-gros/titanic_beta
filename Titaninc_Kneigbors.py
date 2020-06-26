# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 07:58:58 2020

@author: dariu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 12:30:58 2020

@author: dariu
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Import dataset
df = pd.read_csv("train.csv")
#Check which categories have missing values
print(df.isna().sum())
df.head()

#Separate independent and dependent variable in training set
X = df.loc[:, ['Pclass', 'Sex']]
y = df.iloc[:,1:2].values
  #return 1d shape array

#import test set whith the values we need to predict
values_to_predict = pd.read_csv('test.csv')
values_to_predict.head()
#select only p class and sex category
values_to_predict = values_to_predict.loc[:,['Pclass', 'Sex']]

#remove age column

#Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#encoding data
#passthrough so other type of data are not being processed categorically 

from sklearn.preprocessing import OneHotEncoder
hot_enc = OneHotEncoder()
from sklearn.compose import make_column_transformer
column_tran = make_column_transformer((OneHotEncoder(), ['Pclass', 'Sex']))
X_train = np.array(column_tran.fit_transform(X_train))
X_test = np.array(column_tran.fit_transform(X_test))
values_to_predict = np.array(column_tran.fit_transform(values_to_predict))

#create logistic regression model
from sklearn.neighbors import KNeighborsClassifier 
classifier = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski')
classifier.fit(X_train, y_train)

y = y.ravel()
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_real_pred = classifier.predict(values_to_predict)

#convert vector to 2d array
y_real_pred = y_real_pred.reshape(len(y_real_pred), 1)

#join passenger id and predictions
#convert to data frame
passenger_ID = pd.read_csv('test.csv')
passenger_ID = passenger_ID.iloc[:, 0:1].values

pred_table = np.concatenate((passenger_ID, y_real_pred), axis = 1)
pred_converted = pd.DataFrame(data= pred_table ,columns=["PassengerId", "Survived"])
pred_to_csv = pred_converted.to_csv('KNN_prediction_Titanic.csv', encoding='utf-8', index=False)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

y_train = y_train.ravel()

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier, X_train, y_train, cv = 10)
accuracies.mean()
accuracies.std()
