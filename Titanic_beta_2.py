# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:19:54 2020
Titanic beta 2
Data Exploration & Data Cleaning of Titanic Dataset improved                     
@author: dariu
"""

import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('train.csv')
df.head()
df.describe() #891 data points
#drop dependent variable
X = df.drop(['Survived'], axis=1)
#create independent variable
y = df.iloc[:,1:2].values

#Check the correlation between features before multivariate outlier analysis
plt.figure(figsize= (10,10), dpi=100)
sns.heatmap(X.corr())
#no high correlation detected


#ticket and name have no significance for survival prediction. drop them
#keep passenger id, we are gonna need it later
X = X.drop(['PassengerId','Ticket', 'Name'], axis = 1)

X.isna().sum()
#missing points : 177 Age, Cabin 687, Embarked 2
#drop cabin column
X = X.drop(['Cabin'], axis = 1)

#replace missing values of embarked by its mode
X['Embarked'].value_counts()
X['Embarked'].fillna('S', inplace = True)
X.isna().sum() #no null values anymore, okay good

#we can replace age by mean of age 
#check age distribution given class : use boxplot   
sns.boxplot(x = 'Pclass', y = 'Age', data = X, palette = 'winter')

#function to replace missing value by age given class
mean_age = X['Age'].mean()
X['Age'].fillna(mean_age, inplace = True)


#check for outliers for numerical columns: use, boxplot, z score and interquartile range
sns.boxplot(x=X['Age'])
sns.boxplot(x = X['SibSp'])
sns.boxplot(x=X['Parch'])
sns.boxplot(x=X['Fare'])

#encode categorical data
embark = pd.get_dummies(X['Embarked'], drop_first = True)
sex = pd.get_dummies(X['Sex'], drop_first = True)
pclass = pd.get_dummies(X['Pclass'], drop_first = True)

training_categorized = pd.concat([embark, sex, pclass], axis = 1)

#remove data outside of 3std
#table of only numerical values
#drop missing values of Age

X_without_category = X.drop(['Embarked', 'Sex', 'Pclass'], axis = 1)
X_categorized_outliers = pd.concat([X_without_category, training_categorized], axis = 1)
z = np.abs(stats.zscore(X_without_category))
print(z)

threshold = 3
print (np.where(z > 3))

X_without_outliers = X_categorized_outliers[(z < 3).all(axis=1)]
X_without_outliers.shape

#overwrite X for cleaner future input
X = X_without_outliers   #independent variables
y = y[(z<3).all(axis=1)] #dependent variable
y.shape

#no feature scaling since xgb is a tree based model
#######################################################################################################

from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV 
from xgboost.sklearn import XGBClassifier
import xgboost as xgb


#create a classifier with default values
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#fit data to classifier
y.ravel() #convert 1D array
xgb1.fit(X_train, y_train)
y_pred = xgb1.predict(X_test)

#confusion matrix to hget accuracy
from sklearn.metrics import confusion_matrix   
cm = confusion_matrix(y_test, y_pred)
print(cm)

#using kfold to get accuracy of model via mean and std
accuracies_xgb1 = cross_val_score(xgb1, X_train, y_train, cv = 10)
accuracies_xgb1.mean()
accuracies_xgb1.std()

#########################################################################################################


#improve model with grid_search by optimizing parameters
#1st step: optimize max_depth, min child weight
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X_train,y_train)

gsearch1.best_params_, gsearch1.best_score_

#refine grid_search to check max depth
param_test2 = {'max_depth':[6,7,8], 'min_child_weight': [4,5,6]}

gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch2.fit(X_train,y_train)
gsearch2.best_params_, gsearch2.best_score_

#max depth 8, min_child_weight : 5 best paramters
#now tune gamma, this parameter was tuned two times, this is the smaller range.

param_test3 = {'gamma' : [i/10 for i in range(0,5)]}

gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=8,
 min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch3.fit(X_train, y_train)
gsearch3.best_params_, gsearch3.best_score_

#best is gamma = 0.1
#tune subsample and colsample tree

param_test4 = { 'subsample':[i/10.0 for i in range(6,10)],'colsample_bytree':[i/10.0 for i in range(6,10)]}

gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch4.fit(X_train, y_train)
gsearch4.best_params_, gsearch4.best_score_

#colsample_tree 0.9  subsample 0.9

param_test5 = { 'subsample':[i/100.0 for i in range(85,95)],
                             'colsample_bytree':[i/100.0 for i in range(85,95)]}

gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch5.fit(X_train, y_train)
gsearch5.best_params_, gsearch5.best_score_

#still colsample tree 0.89 subsample 0.9, subsitute value
#next:tune regularization parameters reg_alpha.
param_test6 = { 'reg_alpha':[1e-5, 1e-2, 1, 10, 100]}

gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.89,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch6.fit(X_train, y_train)
gsearch6.best_params_, gsearch6.best_score_
#best 1e-05, tried other around 1e-06 and 1-04 too

param_test7 = {'learning_rate' : [0.5,0.1,0.01,0.001]}

gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.6, colsample_bytree=0.8, reg_alpha = 1e-05,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test7, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch7.fit(X_train, y_train)
gsearch7.best_params_, gsearch7.best_score_

param_test8 = {'learning_rate' : [0.1,0.05,0.2,0.3,0.4]}

gsearch8 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.6, colsample_bytree=0.8, reg_alpha = 1e-05,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test8, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch8.fit(X_train, y_train)
gsearch8.best_params_, gsearch8.best_score_
#0.3 for learning rate the best

#tune estimators
param_test9 = {'n_estimators': range (100,1000,100)}

gsearch9 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.3, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.6, colsample_bytree=0.8, reg_alpha = 1e-05,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test9, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch9.fit(X_train, y_train)
gsearch9.best_params_, gsearch9.best_score_
#n_estimators = 200

#will have to think of a way to automate grid search via function to save time
param_test10 = {'n_estimators': range (140,260,10)}

gsearch10 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.3, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.6, colsample_bytree=0.8, reg_alpha = 1e-05,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test10, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch10.fit(X_train, y_train)
gsearch10.best_params_, gsearch10.best_score_
#250

########################################################################################################

#creating final object with parameters optimized
xgb_final = XGBClassifier(
        learning_rate =0.3, 
        n_estimators=250, 
        max_depth=4,
        min_child_weight=6, 
        gamma=0, 
        subsample=0.6,
        colsample_bytree=0.8,
        reg_alpha = 1e-05,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

xgb_final.fit(X_train, y_train)
#y.ravel()

y_pred = xgb_final.predict(X_test)

from sklearn.metrics import confusion_matrix   #Additional scklearn functions
cm = confusion_matrix(y_test, y_pred)
print(cm)

##using kfold to get accuracy of model via mean and std
accuracies = cross_val_score(xgb_final, X_train, y_train, cv = 10)
accuracies.mean()
accuracies.std()

############################################################################################################

#predict binary outcome of new file

test_data = pd.read_csv('test.csv')
test_data['Embarked'].fillna('S', inplace = True)
test_data = test_data.drop(['Cabin'], axis = 1)
test_data = test_data.drop(['PassengerId', 'Ticket', 'Name'], axis = 1)
#function to replace missing value by age given class
test_mean_age = test_data['Age'].mean()
test_data['Age'].fillna(test_mean_age, inplace = True)

test_data['Fare'].isna().sum()
test_mean_fare = test_data['Fare'].mean()
test_data['Fare'].fillna(test_mean_fare, inplace = True)

test_embark = pd.get_dummies(test_data['Embarked'], drop_first = True)
test_sex = pd.get_dummies(test_data['Sex'], drop_first = True)
test_pclass = pd.get_dummies(test_data['Pclass'], drop_first = True)
test_data_categorized = pd.concat([test_embark, test_sex, test_pclass], axis = 1)

test_data_without_category = test_data.drop(['Embarked', 'Sex', 'Pclass'], axis = 1)
test_data_categorized_outliers = pd.concat([test_data_without_category, test_data_categorized], axis = 1)
test_data_categorized_outliers.shape
#data given without the outcome, all we have are independent variables
test_y_pred = xgb_final.predict(test_data_categorized_outliers)

#CONVERT to 1d dimension
test_y_pred = test_y_pred.reshape(len(test_y_pred), 1)

#get passenger ID from reduced outliers dataframe
test_ID = test_data.iloc[:, 0:1]
#export as csv
#join passenger id and predictions
#convert to data frame
pred_table = np.concatenate((test_ID, test_y_pred), axis = 1)
pred_converted = pd.DataFrame(data= pred_table, columns=["PassengerId", "Survived"])
pred_to_csv = pred_converted.to_csv('XGB_V2_Titanic.csv', encoding='utf-8', index=False)











 
