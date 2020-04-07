#!/usr/bin/env python
# coding: utf-8

# In[292]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import math
sns.set_style('whitegrid')

import os

pd.options.mode.chained_assignment = None # Warning for chained copies disabled

from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter(action='ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[293]:


data = pd.read_csv("FlightDelays.csv")
print(data.isnull().sum())
data1 = data


# In[295]:


target = data1[['Flight Status']]
c=data1.drop(['Flight Status','FL_DATE'], axis=1, inplace=False)


# In[296]:


label_encoder = preprocessing.LabelEncoder() 
train=c

train['DEST']= label_encoder.fit_transform(train['DEST'])  
train['ORIGIN']= label_encoder.fit_transform(train['ORIGIN'])  
train['CARRIER']= label_encoder.fit_transform(train['CARRIER']) 
train['TAIL_NUM']= label_encoder.fit_transform(train['TAIL_NUM']) 


# In[299]:


feature=list(train.columns.values)


# In[300]:


f,ax=plt.subplots(figsize=(10,7))
sns.heatmap(train.corr(), annot=True, fmt = ".2f", cmap='viridis')
plt.title('Correlation between features', fontsize=10, weight='bold' )
plt.show()


# In[302]:


t=np.array(target)
t[t=='delayed']=1
t[t=='ontime']=0
t = t.astype('int64')


# In[303]:


x=train
y=t


# In[304]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .4, random_state=0)


# In[307]:


from sklearn.linear_model import LogisticRegression

lr_c=LogisticRegression( C=1, max_iter=50000,
                     penalty='l2',
                   random_state=None, solver='lbfgs', verbose=2,
                   warm_start=True)
lr_c.fit(x_train,y_train.ravel())
coef=lr_c.predict_proba(x_test)
lr_pred=lr_c.predict(x_test)
lr_ac=accuracy_score(y_test.ravel(), lr_pred)
print('LogisticRegression_accuracy test:',lr_ac)
print("AUC",roc_auc_score(y_test.ravel(), lr_pred))


# In[308]:


coef = lr_c.coef_[0]

