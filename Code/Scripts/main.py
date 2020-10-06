# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:14:26 2017

@author: Madhu
"""
# Import libraries necessary for this project
import sklearn
#import networkx as nx
import numpy  as np
import pandas as pd
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt
from pandas import compat

compat.PY3 = True
print ("-----------------------------------------------------------------------")
print('The scikit-learn version is {}.'.format(sklearn.__version__))

#load functions from 
from projectFunctions import loadData, exploreData, missingValues, transformData

path = r'C:\Users\pmspr\Documents\HS\MS\Sem 3\EECS 731\Week 6\HW\Git\EECS-731-Project-4\Data'
filename = "nfl_games.csv"
data_raw = loadData(path,filename)
data = data_raw.drop(['neutral','playoff','date','season','result1'], axis = 1)
#data.rename(columns={'elo_prob1':'target'},inplace=True)
teams = data['team1'].unique()

data_ct = data.groupby(['team1','team2'], as_index=False).count()
data_ct = pd.DataFrame(data_ct, columns=['team1','team2','score1'])
data_ct.rename(columns={'score1':'Count'},inplace=True)
data = pd.merge(data,data_ct, on=['team1','team2'], how='inner')

#Check the missing values
misVal, mis_val_table_ren_columns = missingValues(data)
print(mis_val_table_ren_columns.head(20))

t1 = pd.DataFrame(data, columns=['team1','score1'])
t1 = t1.groupby(['team1'], as_index=False).count()

t2 = pd.DataFrame(data, columns=['team2','score2'])
t2 = t2.groupby(['team2'], as_index=False).count()

tp = pd.DataFrame([], columns=['team','points'])
rows = []
dicts = {}

for team in teams:
    ind = np.where(t1['team1'] == team)
    if len(ind[0]) > 0:
        val = float(t1['score1'].iloc[ind])
        s1 =  0 if (val != val)  else val
    ind = np.where(t2['team2'] == team)
    if len(ind[0]) > 0:
        val = float(t2['score2'].iloc[ind])
        s2 = 0 if (val != val) else val
    dicts[team] = s1 + s2

tp['team'] = dicts.keys()
tp['points'] = dicts.values()

from projectFunctions import barPlot, numCount, corrPlot, splitData
#barPlot(tp['team'], tp['points'],'Teams','Scores','Points by team')
#numCount(data,'score1','score2','Score distribution')
#numCount(data,'elo1','elo2','elo distribution')

#Remove categorical columns for correlation heatmap 
data_corr = data.drop(['team1','team2'], axis = 1)
corr = data_corr.corr()
#corrPlot(corr)

features,target = exploreData(data)
features,target = transformData(features,target)

X_train, X_test, y_train, y_test = splitData(features,target,0.3)

from projectFunctions import lineReg, sdgReg, ridgeReg, lassoReg

res_pd = pd.DataFrame([], columns=['Model','AccTrain','AccTest','TrainTime','PredTime'])

results,clf_fit_train = lineReg(X_train, X_test, y_train, y_test)

print "-----------------------------------------------------------------------"
print "Times for Training, Prediction: %.5f, %.5f" %(results['train_time'], results['pred_time'])     
print "Accuracy for Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test'])     
print "-----------------------------------------------------------------------"
res_pd.loc[0,'AccTrain'] = results['acc_train']
res_pd.loc[0,'AccTest'] = results['acc_test']
res_pd.loc[0,'TrainTime'] = results['train_time']
res_pd.loc[0,'PredTime'] = results['pred_time']
res_pd.loc[0,'Model'] = 'linear'

results,clf_fit_train = sdgReg(X_train, X_test, y_train, y_test)

print "-----------------------------------------------------------------------"
print "Times for Training, Prediction: %.5f, %.5f" %(results['train_time'], results['pred_time'])     
print "Accuracy for Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test'])     
print "-----------------------------------------------------------------------"
res_pd.loc[1,'AccTrain'] = results['acc_train']
res_pd.loc[1,'AccTest'] = results['acc_test']
res_pd.loc[1,'TrainTime'] = results['train_time']
res_pd.loc[1,'PredTime'] = results['pred_time']
res_pd.loc[1,'Model'] = 'SDG'

results,clf_fit_train = ridgeReg(X_train, X_test, y_train, y_test)

print "-----------------------------------------------------------------------"
print "Times for Training, Prediction: %.5f, %.5f" %(results['train_time'], results['pred_time'])     
print "Accuracy for Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test'])     
print "-----------------------------------------------------------------------"
res_pd.loc[2,'AccTrain'] = results['acc_train']
res_pd.loc[2,'AccTest'] = results['acc_test']
res_pd.loc[2,'TrainTime'] = results['train_time']
res_pd.loc[2,'PredTime'] = results['pred_time']
res_pd.loc[1,'Model'] = 'Ridge'

results,clf_fit_train = lassoReg(X_train, X_test, y_train, y_test)

print "-----------------------------------------------------------------------"
print "Times for Training, Prediction: %.5f, %.5f" %(results['train_time'], results['pred_time'])     
print "Accuracy for Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test'])     
print "-----------------------------------------------------------------------"
res_pd.loc[3,'AccTrain'] = results['acc_train']
res_pd.loc[3,'AccTest'] = results['acc_test']
res_pd.loc[3,'TrainTime'] = results['train_time']
res_pd.loc[3,'PredTime'] = results['pred_time']
res_pd.loc[3,'Model'] = 'Lasso'

plt.figure(figsize=(20,5))
sns.barplot(res_pd['Model'],res_pd['AccTrain'], alpha=0.8)
plt.title('title')
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.show()

features.to_csv('test.csv',index=False)