# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 19:06:37 2021

@author: Wilson
"""
from sklearn.svm import SVC, LinearSVC

model1 = SVC()
model2 = LinearSVC()

from sklearn.datasets import load_breast_cancer
X, y = data = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8675309)

import pandas as pd
from sklearn import model_selection
from sklearn.metrics import classification_report

scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
target_names = ['malignant', 'benign']
results = []

def run_exps(X_train: pd.DataFrame , y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, model) -> pd.DataFrame:
     kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
     cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
     clf = model.fit(X_train, y_train)
     y_pred = clf.predict(X_test)
        
     print(model)
     print(classification_report(y_test, y_pred, target_names=target_names))
     
     results.append(cv_results)
     
     return results

run_exps(X_train, y_train, X_test, y_test, model1)