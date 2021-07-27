# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 19:06:37 2021

@author: Wilson, Michael
"""
from sklearn.datasets import load_breast_cancer
X, y = data = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8675309)


# Deep learning Models
# Binary Classification with Sonar Dataset: Baseline
# from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=30, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
estimator2 = KerasClassifier(build_fn=baseline_model, epochs=2, batch_size=1, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)

# # ------------------------------------------------------------------------------------ END OF MODEL

import pandas as pd
from sklearn import model_selection
from sklearn.metrics import classification_report

scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
target_names = ['malignant', 'benign']
results = []

def DIFF_CHECKER(X_train: pd.DataFrame , y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, DNNF, DNNC) -> pd.DataFrame:
      kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
      cv_results = model_selection.cross_validate(DNNF, X_train, y_train, cv=kfold, scoring=scoring)
      clf = DNNF.fit(X_train, y_train)
      y_pred1= DNNF.predict(X_test)
       
      cv_results = model_selection.cross_validate(DNNC, X_train, y_train, cv=kfold, scoring=scoring)
      clf = DNNC.fit(X_train, y_train)
      y_pred2= DNNC.predict(X_test)
        
      print("DNNF")
      print(classification_report(y_test, y_pred1, target_names=target_names))
      
        
      print("DNNC")
      print(classification_report(y_test, y_pred2, target_names=target_names))
     
      results.append(cv_results)
     
      return results

DIFF_CHECKER(X_train, y_train, X_test, y_test, estimator2, estimator)