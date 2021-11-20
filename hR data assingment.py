# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:42:58 2021

@author: khanderao patale
"""

import pandas as pd

dataset =pd.read_csv('C:/khanderao/HR_Data.csv')

#handling categorical var
dataset.info()
pd.get_dummies(dataset['role'])
pd.get_dummies(dataset['role'],drop_first= True)
r_dummy = pd.get_dummies(dataset['role'], drop_first= True)
r_dummy.head(5)

pd.get_dummies(dataset['salary'])
pd.get_dummies(dataset['salary'], drop_first= True)
s_dummy = pd.get_dummies(dataset['salary'], drop_first= True)
s_dummy.head(5)

#concating r_dummy and s_dummy with dataset#

dataset = pd.concat([dataset, r_dummy, s_dummy,], axis=1)
dataset.head(5)

#dataset.info()

dataset.drop(['role','salary',],axis=1,inplace=True)



#obtaining DV and IV var
x=dataset.drop('left',axis=1)
y=dataset['left']

# spliting dataset trainig and testing

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

#fitting log regression 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
#from sklearn.linear_model import LogisticRegression
#logmodel = LogisticRegression()
#ogmodel.fit(x_train, y_train)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)

prediction = logmodel.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, prediction)


#calculate the cof
print(logmodel.coef_)

#catlcutate intercept

print(logmodel.intercept_)



dataset1=dataset
dataset1.head(5)

import statsmodels.api as sm


import numpy as nm
x = nm.append(arr = nm.ones((14999,1)).astype(int),values=x, axis=1)
#x1 = nm.append(arr = nm.ones((14999,1)).astype(int), Value=x, axis=1)

x_opt=x[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,]]

regression_OLS=sm.OLS(endog=y,exog=x_opt).fit()

regression_OLS.summary()

x_opt=x[:,[0,1,2,3,4,5,6,7,8,10,17,18,]]

regression_OLS=sm.OLS(endog=y,exog=x_opt).fit()

regression_OLS.summary()

from sklearn.model_selection import train_test_split
x_BE_train, x_BE_test, y_BE_train, y_BE_test = train_test_split(x_opt, y, test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_BE_train, y_BE_train)

prediction = logmodel.predict(x_BE_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_BE_test,prediction)

logmodel.coef_

logmodel.intercept_
































































































































































