#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 21:26:45 2020

@author: s.p.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
dataset['DESCR']
dataset['data']
dataset['feature_names']
len(dataset['feature_names'])
dataset['filename']
dataset['target']
dataset['target_names']

#Import Data and define the variables
variables = np.c_[dataset['data'], dataset['target']]
data_label = np.append(dataset['feature_names'], ['target'])
df_cancer = pd.DataFrame(variables, columns = data_label)
x = df_cancer.iloc[:,:-1].values
y = df_cancer.iloc[:,-1:].values

#Data Visualizaion
df_cancer.head()
df_cancer.info()
df_cancer.describe()
sns.pairplot(df_cancer, vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'], hue = 'target')
sns.countplot(df_cancer['target'])
sns.heatmap(df_cancer.corr(), annot = True)

#Split data into test and training sets
from sklearn.model_selection import train_test_split
x_train, x_test , y_train, y_test = train_test_split(x,y,test_size = .25, random_state = 0)
#Apply Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


'''---Building a Classification Model using SVM---'''
from sklearn.svm import SVC
classifier = SVC(C= 10, gamma = 1 ,kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print("Model Accuracy: ", accuracy_score(y_test, y_pred))


#Refine Prediction
param_grid =[{'C':[0.1, 1, 10, 100], 'gamma':[1, 0.1, 0.01, 0.001], 'kernel':['rbf'],'random_state':[0]},
             {'C':[0.1, 1, 10, 100], 'gamma':[1, 0.1, 0.01, 0.001], 'kernel':['linear'],'random_state':[0]}]
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid,scoring = 'accuracy',refit = True, verbose = 5)

#Fit Refined prediction
grid.fit(x_train, y_train)
y_refined = grid.predict(x_test)

#Evaluate Model
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_refined)
sns.heatmap(cm, annot = True)
plt.title('SVM')
print("Highest Grid Score: ",grid.best_score_)
print("Best Paramerters: ",grid.best_params_)
print("Refined Model Accuracy: ", accuracy_score(y_test, y_refined))



'''---Building a Classification Model using Random Forest---'''
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print("Model Accuracy: ", accuracy_score(y_test, y_pred))

#Refine Prediction
param_grid =[{'n_estimators':[5.10,20,50,100,200], 'criterion':['gini'], 'random_state':[0]},
             {'n_estimators':[5.10,20,50,100,200], 'criterion':['entropy'], 'random_state':[0]}]
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(RandomForestClassifier(), param_grid,scoring = 'accuracy',refit = True, verbose = 5)

#Fit Refined prediction
grid.fit(x_train, y_train)
y_refined = grid.predict(x_test)

#Evaluate Model
cm = confusion_matrix(y_test, y_refined)
sns.heatmap(cm, annot = True)
plt.title('Random Forest')
print("Highest Grid Score: ",grid.best_score_)
print("Best Paramerters: ",grid.best_params_)
print("Refined Model Accuracy: ", accuracy_score(y_test, y_refined))



'''---Building a Classification Model using Naive Bayes---''' 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

#Evaluate Model
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True)
plt.title('Naive Bayes')
print("Model Accuracy: ", accuracy_score(y_test, y_pred))


'''---Building a Classification Model using KNearest Neighbors---''' 
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print("Model Accuracy: ", accuracy_score(y_test, y_pred))


#Refine Prediction
param_grid =[{'n_neighbors':[5.10,15,20,25,50,100], 'metric':['minkowski'], 'p':[1,2,3,4,5]},
             {'n_neighbors':[5.10,15,20,25,50,100], 'metric':['seuclidean'], 'p':[1,2,3,4,5]},
             {'n_neighbors':[5.10,15,20,25,50,100], 'metric':['euclidean'], 'p':[1,2,3,4,5]}
             ]
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(KNeighborsClassifier(), param_grid,scoring = 'accuracy',refit = True, verbose = 5)

#Fit Refined prediction
grid.fit(x_train, y_train)
y_refined = grid.predict(x_test)

#Evaluate Model
cm = confusion_matrix(y_test, y_refined)
sns.heatmap(cm, annot = True)
plt.title('KNN')
print("Highest Grid Score: ",grid.best_score_)
print("Best Paramerters: ",grid.best_params_)
print("Refined Model Accuracy: ", accuracy_score(y_test, y_refined))







