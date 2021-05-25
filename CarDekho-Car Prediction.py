#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

df = pd.read_csv('car data.csv')
final_ds=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_ds['Current_Year']=2021
final_ds['no_year']=final_ds['Current_Year']-final_ds['Year']
final_ds.drop(['Year'],axis=1,inplace=True)
final_ds.drop(['Current_Year'],axis=1,inplace=True)
final_ds=pd.get_dummies(final_ds,drop_first=True)

### independent and dependent features
X=final_ds.iloc[:,1:]
y=final_ds.iloc[:,0]

### feature importance
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

### hyperparameters
#Randomized search CV
#no of trees in the random forest
n_estimators =[int(x) for x in np.linspace(start=100, stop =1200, num=12)]
print(n_estimators)
#of features at every split
max_features=['auto','sqrt']
#max number of levels in trees
max_depth=[int(x) for x in np.linspace(start=5,stop=30,num=6)]
#minimum number of samples required to split a node
min_samples_split=[2,5,10,15,100]
#min number of samples required at each leaf node
min_samples_leaf=[1,2,5,10]

#create the random grid
random_grid={'n_estimators': n_estimators,
             'max_features': max_features,
             'max_depth': max_depth,
             'min_samples_split': min_samples_split,
             'min_samples_leaf': min_samples_leaf}

print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf=RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(X_train,y_train)

predictions=rf_random.predict(X_test)

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)
file.close()
