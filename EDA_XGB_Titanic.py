# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:21:25 2020

@author: dariu
"""

#Titanic EDA
import pandas as pd
import numpy as np 
import seaborn as sns


#Import training set
dataset = pd.read_csv('train.csv')
dataset.head()
dataset.isna().sum()

#EAD: Visualization
#female/male survived
sns.set_style('whitegrid')
sns.countplot(x = 'Survived', hue = 'Sex', data = dataset, palette = 'winter')

#nb survived given classes
sns.set_style()
sns.countplot(x = 'Survived', hue = 'Pclass', data = dataset, palette = 'winter')

#sns.set_style()
#sns.countplot(x = 'Survived', hue = 'Age', data = dataset, palette = 'winter')

#distribution of age, drop missing values to visualize distribution
sns.distplot(dataset['Age'].dropna(), kde = False, color = 'darkred', bins = 40)

#relation between age and class: we can see higher classes are older
sns.boxplot(x = 'Pclass', y = 'Age', data = dataset, palette = 'winter')


#######################################################################################


#embarked only missing two values replace them by the most frequent
dataset['Embarked'].value_counts()
dataset['Embarked'].fillna('S', inplace = True)
#check if embarked has no null values anymore, okay good
dataset.isna().sum()

#use heatmap to visualize null data
sns.heatmap(dataset.isna(), yticklabels = 'False', cbar = 'False', cmap = 'viridis')
dataset = dataset.drop(['Cabin'], axis = 1)

#passenger id, ticket and name have no significance for survival prediction. drop them
dataset = dataset.drop(['PassengerId', 'Ticket', 'Name'], axis = 1)


#getmean value of age for each class and replace na values for age if working with
#a non based tree classifier

#encode categorical as dummies
data_without_category = dataset.drop(['Embarked', 'Sex'], axis = 1)
embark = pd.get_dummies(dataset['Embarked'], drop_first = False)
sex = pd.get_dummies(dataset['Sex'], drop_first = False)

training_categorized = pd.concat([data_without_category, embark, sex], axis = 1)
training_categorized.head()

X = training_categorized.iloc[:, 1:11].values
y = training_categorized.iloc[:,0:1].values

#no need for feature scaling since using xgboost classifier

#######################################################################################


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
y.ravel()
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

#######################################################################################################

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
param_test2 = {'max_depth':[3,4,5,6,7], 'min_child_weight': [4,5,6]}

gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch2.fit(X_train,y_train)
gsearch2.best_params_, gsearch2.best_score_

#max depth 4, min_child_weight : 6 best paramters
#now tune gamma, this parameter was tuned two times, this is the smaller range.

param_test3 = {'gamma' : [i/10 for i in range(0,5)]}

gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch3.fit(X_train, y_train)
gsearch3.best_params_, gsearch3.best_score_

#best is gamma = 0
#tune subsample and colsample tree

param_test4 = { 'subsample':[i/10.0 for i in range(6,10)],'colsample_bytree':[i/10.0 for i in range(6,10)]}

gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch4.fit(X_train, y_train)
gsearch4.best_params_, gsearch4.best_score_

#colsample_tree 0.8  subsample 0.6

param_test5 = { 'subsample':[i/100.0 for i in range(55,65)],
                             'colsample_bytree':[i/100.0 for i in range(75,85)]}

gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch5.fit(X_train, y_train)
gsearch5.best_params_, gsearch5.best_score_

#still colsample tree 0.8 subsample 0.6, subsitute value
#next:tune regularization parameters gamma.
param_test6 = { 'reg_alpha':[1e-5, 1e-2, 1, 10, 100]}

gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.6, colsample_bytree=0.8,
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

param_test8 = {'learning_rate' : [0.1,0.09,0.11]}

gsearch8 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.6, colsample_bytree=0.8, reg_alpha = 1e-05,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test8, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch8.fit(X_train, y_train)
gsearch8.best_params_, gsearch8.best_score_
#0.1 for learning rate the best

#tune estimators
param_test9 = {'n_estimators': range (100,1000,100)}

gsearch9 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.6, colsample_bytree=0.8, reg_alpha = 1e-05,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test9, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch9.fit(X_train, y_train)
gsearch9.best_params_, gsearch9.best_score_
#n_estimators = 300

#3r iteratoion,last one said n_estimators best at 280 but because redundant code, did not keep it. 
#will have to think of a way to automate that code via function to save time
param_test10 = {'n_estimators': range (270,290,5)}

gsearch10 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.6, colsample_bytree=0.8, reg_alpha = 1e-05,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test10, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch10.fit(X_train, y_train)
gsearch10.best_params_, gsearch10.best_score_
#280 best

#some additional visualization to provide : importance of features when performing grid search using matplotlib

########################################################################################################

#creating final onject with parameters optimized
xgb_final = XGBClassifier(
        learning_rate =0.1, 
        n_estimators=280, 
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
#give accuracy of 82% lower than when testing model in gridsearch at 86.5%.
#might have slight overfitting - need to look at other regularization parameters for xgb classifier object

############################################################################################################

#predict binary outcome of new file

test_data = pd.read_csv('test.csv')
test_data['Embarked'].fillna('S', inplace = True)
test_data = test_data.drop(['Cabin'], axis = 1)
test_data = test_data.drop(['PassengerId', 'Ticket', 'Name'], axis = 1)
test_data_without_category = test_data.drop(['Embarked', 'Sex'], axis = 1)
test_embark = pd.get_dummies(test_data['Embarked'], drop_first = False)
test_sex = pd.get_dummies(test_data['Sex'], drop_first = False)

#data given without the outcome, all we have are independent variables
X_test_data = pd.concat([test_data_without_category, 
                                       test_embark,
                                       test_sex], axis = 1)

X_test_data = X_test_data.iloc[:,:].values
test_y_pred = xgb_final.predict(X_test_data)

#CONVERT 2D
test_y_pred = test_y_pred.reshape(len(test_y_pred), 1)

#export as csv
#join passenger id and predictions
#convert to data frame

passenger_ID = pd.read_csv('test.csv')
passenger_ID = passenger_ID.iloc[:, 0:1].values

pred_table = np.concatenate((passenger_ID, test_y_pred), axis = 1)
pred_converted = pd.DataFrame(data= pred_table ,columns=["PassengerId", "Survived"])
pred_to_csv = pred_converted.to_csv('XGB_Titanic.csv', encoding='utf-8', index=False)



