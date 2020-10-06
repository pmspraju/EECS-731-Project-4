# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:20:49 2017

@author: Madhu
"""
import os
#import sys
import time
import pandas as pd
import numpy  as np
#import nltk
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199
#nltk.download('punkt')

#Function to load the data
def loadData(path,filename):
    try:
             files = os.listdir(path)
             for f in files:
                 if f == filename:
                     data = pd.read_csv(os.path.join(path,f))
                     return data
            
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

#Function to explore the data
def exploreData(data):
    try:
           #Total number of records                                  
           rows = data.shape[0]
           cols = data.shape[1]    
           
           #separate features and target
           drop_col = ['elo_prob1']
           features = data.drop(drop_col, axis = 1)
           target = data[drop_col]
          
           # Print the results
           print ("-----------------------------------------------------------------------")
           print ("Total number of records: {}".format(rows))
           print ("Total number of features: {}".format(cols))
           print ("-----------------------------------------------------------------------")
           
           return features,target
           
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

def missingValues(data):
    try:
           # Total missing values
           mis_val = data.isnull().sum()
         
           # Percentage of missing values
           mis_val_percent = 100 * mis_val / len(data)
           
           # Make a table with the results
           mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
           
           # Rename the columns
           mis_val_table_ren_columns = mis_val_table.rename(
           columns = {0 : 'Missing Values', 1 : '% of Total Values'})
           mis_val_table_ren_columns.head(4 )
           # Sort the table by percentage of missing descending
           misVal = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
                   '% of Total Values', ascending=False).round(1)
                     
           return misVal, mis_val_table_ren_columns

    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

def transformData(features,target):
    try:    
        # TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
        #features_final = pd.get_dummies(features_log_minmax_transform)
        
        features_log_minmax_transform = pd.DataFrame(data = features)
        enc = LabelEncoder()
        features_log_minmax_transform['team1'] = enc.fit_transform(features_log_minmax_transform['team1'])
        features_log_minmax_transform['team2'] = enc.fit_transform(features_log_minmax_transform['team2'])
        scaler = MinMaxScaler() # default=(0, 1)
        numerical = ['team1','team2','elo1','elo2','score1','score2','Count']
        features_log_minmax_transform[numerical] = scaler.fit_transform(features[numerical])
               
        target_f = scaler.fit_transform(target)
        features_f = features_log_minmax_transform
        
        return features_f, target_f
        
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

#split the data in to train and test data
def splitData(features,target, testsize):
    try:
        # Split the 'features' and 'income' data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features,
                                                            target,
                                                            test_size = testsize, 
                                                            random_state = 1)

        # Show the results of the split
        print ("Features training set has {} samples.".format(X_train.shape[0]))
        print ("Features testing set has {} samples.".format(X_test.shape[0]))
        print ("Target training set has {} samples.".format(y_train.shape[0]))
        print ("Target testing set has {} samples.".format(y_test.shape[0]))
        print ("-----------------------------------------------------------------------")
        return X_train, X_test, y_train, y_test
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
           
def barPlot(l1,l2,xd,yd,title):
    try:
        plt.figure(figsize=(20,5))
        sns.barplot(l1, l2, alpha=0.8)
        plt.title(title)
        plt.ylabel(yd, fontsize=12)
        plt.xlabel(xd, fontsize=12)
        plt.xticks(rotation=90)
        plt.show()
    except Exception as ex:
        print "-----------------------------------------------------------------------"
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message
        
def numCount(data,f1,f2,title):
    try:
        plt.figure(figsize = (12, 6))
        sns.kdeplot(data[f1], label='Team 1')
        sns.kdeplot(data[f2], label='Team 2')
        plt.title(title)
        plt.legend();
    except Exception as ex:
        print "-----------------------------------------------------------------------"
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message      
        
def corrPlot(corr):
    try:
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))
        
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
    except Exception as ex:
        print "-----------------------------------------------------------------------"
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message      
        
def gridSearch(X_train, X_test, y_train, y_test,clf):
    try:
        params = {}
         
        scoring_fnc = make_scorer(r2_score)
        learner = GridSearchCV(clf,params,scoring=scoring_fnc)
        results = {}
         
        start_time = time.clock()
        grid = learner.fit(X_train,y_train)
         
        end_time = time.clock()
        results['train_time'] = end_time - start_time
        clf_fit_train = grid.best_estimator_
        start_time = time.clock()
         
        clf_predict_train = clf_fit_train.predict(X_train)
        clf_predict_test = clf_fit_train.predict(X_test)
         
        end_time = time.clock()
        results['pred_time'] = end_time - start_time  
         
        results['acc_train'] = r2_score(y_train, clf_predict_train)
        results['acc_test']  = r2_score(y_test, clf_predict_test)
        
        return results,clf_fit_train      
    except Exception as ex:
           print "-----------------------------------------------------------------------"
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print message

def lineReg(X_train, X_test, y_train, y_test):
    try:
         clf = linear_model.LinearRegression()
         results,clf_fit_train = gridSearch(X_train, X_test, y_train, y_test,clf)
         return results,clf_fit_train
    except Exception as ex:
           print "-----------------------------------------------------------------------"
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print message

def sdgReg(X_train, X_test, y_train, y_test):
    try:
         clf = linear_model.SGDRegressor()
         results,clf_fit_train = gridSearch(X_train, X_test, y_train, y_test,clf)
         return results,clf_fit_train
    except Exception as ex:
           print "-----------------------------------------------------------------------"
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print message
           
def ridgeReg(X_train, X_test, y_train, y_test):
    try:
         clf = linear_model.Ridge(alpha=1,solver="cholesky")
         results,clf_fit_train = gridSearch(X_train, X_test, y_train, y_test,clf)
         return results,clf_fit_train
    except Exception as ex:
           print "-----------------------------------------------------------------------"
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print message
           
def lassoReg(X_train, X_test, y_train, y_test):
    try:
         clf = linear_model.Lasso(alpha=0.1)
         results,clf_fit_train = gridSearch(X_train, X_test, y_train, y_test,clf)
         return results,clf_fit_train
    except Exception as ex:
           print "-----------------------------------------------------------------------"
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print message