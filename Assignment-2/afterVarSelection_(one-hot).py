#!/usr/bin/env python
# coding: utf-8

# In[530]:


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


# In[531]:


data = pd.read_csv("FlightDelays.csv")
print(data.isnull().sum())


# In[532]:


data1=data[data['Weather'] == 0]
target = data1[['Flight Status']]
c=data1.drop(['Flight Status','FL_DATE'], axis=1, inplace=False)


# In[533]:


c['DELAY']=abs((c['CRS_DEP_TIME'])-(c['DEP_TIME']))


# In[534]:


c2=c.drop(['Weather','DEST', 'DAY_WEEK','DISTANCE','CARRIER','FL_NUM'],axis=1, inplace=False)


# In[535]:


c2.sample(5)


# In[536]:


train=pd.get_dummies(c2)


# In[537]:


features=list(train.columns.values)


# In[538]:


t=np.array(target)
t[t=='delayed']=1
t[t=='ontime']=0
t = t.astype('int64')


# In[539]:


x=train
y=t


# In[540]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .4, random_state=0)


# In[541]:


from sklearn.linear_model import LogisticRegression

lr_c=LogisticRegression( C=1, max_iter=60000,
                     penalty='l2',
                   random_state=None, solver='lbfgs', verbose=2,
                   warm_start=True)
lr_c.fit(x_train,y_train.ravel())
lr_pred=lr_c.predict(x_test)
lr_ac=accuracy_score(y_test.ravel(), lr_pred)
print('LogisticRegression_accuracy test:',lr_ac)
print("AUC",roc_auc_score(y_test.ravel(), lr_pred))


# In[542]:


coef = lr_c.coef_[0]
len(coef)

