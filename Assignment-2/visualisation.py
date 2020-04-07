#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')

pd.options.mode.chained_assignment = None # Warning for chained copies disabled

import warnings
warnings.simplefilter(action='ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[63]:


data = pd.read_csv("FlightDelays.csv")


# In[64]:


num=data['FL_NUM'].value_counts()
num=pd.DataFrame(num)
num=num.reset_index()

carrier= data['CARRIER'].value_counts()
carrier= pd.DataFrame(carrier)
carrier= carrier.reset_index()

dest= data['DEST'].value_counts()
dest= pd.DataFrame(dest)
dest= dest.reset_index()

org= data['ORIGIN'].value_counts()
org= pd.DataFrame(org)
org= org.reset_index()

status= data['Flight Status'].value_counts()
status= pd.DataFrame(status)
status= status.reset_index()

dist=data['DISTANCE'].value_counts()
dist= pd.DataFrame(dist)
dist= dist.reset_index()

weather=data['Weather'].value_counts()
weather= pd.DataFrame(weather)
weather= weather.reset_index()

day= data['DAY_WEEK'].value_counts()
day= pd.DataFrame(day)
day= day.reset_index()

plt.style.use('seaborn')
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,15))
ax1 = plt.subplot2grid((3,2),(0,0))
plt.pie(day.DAY_WEEK,colors=("gold","sienna","plum","deepskyblue","lawngreen", 'crimson', 'pink'), autopct='%2.1f%%',labels=day['index'], shadow=True)
plt.title('Day of Week', fontsize=13, weight='bold' )

ax1 = plt.subplot2grid((3,2),(0,1))
plt.pie(carrier.CARRIER,colors=("cyan","lime", 'orange',"deepskyblue","lawngreen", 'crimson',"gold","sienna"), autopct='%2.1f%%',labels=carrier['index'], shadow=True)
plt.title('Carrier', fontsize=13, weight='bold' )

ax1 = plt.subplot2grid((3,2),(1,0))
plt.pie(dest.DEST,colors=("darkcyan","purple","gold"), autopct='%2.1f%%',labels=dest['index'], shadow=True)
plt.title('Destination of flight', fontsize=13, weight='bold' )

ax1 = plt.subplot2grid((3,2),(1,1))
plt.pie(org.ORIGIN,colors=("deepskyblue","lawngreen", 'crimson'), autopct='%2.1f%%',labels=org['index'], shadow=True)
plt.title('Origin of flight', fontsize=13, weight='bold' )

ax1 = plt.subplot2grid((3,2),(2,0))
plt.pie(weather.Weather,colors=("gold","sienna","plum","deepskyblue","lawngreen", 'crimson'), autopct='%2.1f%%',labels=weather['index'], shadow=True)
plt.title('Weather', fontsize=13, weight='bold' )

ax1 = plt.subplot2grid((3,2),(2,1))
plt.pie(dist.DISTANCE,colors=("gold","sienna","plum","deepskyblue","lawngreen", 'crimson', 'red'), autopct='%2.1f%%',labels=dist['index'], shadow=True)
plt.title('Distance', fontsize=13, weight='bold' )


plt.show()


# In[65]:


f,ax=plt.subplots(figsize=(7,6))
data['DISTANCE'].plot(kind='hist', color='orange')
plt.title('Distribution of travel distances', fontsize=10, weight='bold' )
plt.show()


# In[66]:


sns.catplot(x='DAY_WEEK', kind='count',hue='Flight Status', data=data, palette='ch:.384')
plt.title('Delay per day of week', fontsize=10, weight='bold' )
plt.show()


# In[67]:


sns.catplot(x='FL_DATE', kind='count',hue='Flight Status', data=data, palette='ch:.384')
plt.title('Delay per date', fontsize=10, weight='bold' )
plt.show()


# In[68]:


sns.catplot(x='DISTANCE', kind='count',hue='Flight Status', data=data, palette='ch:.384')
plt.title('Delay per day of month', fontsize=10, weight='bold' )
plt.show()


# In[69]:


sns.catplot(x='Weather', kind='count',hue='Flight Status',height=6, data=data, palette='Paired')
plt.title('Delay with respect to destination', fontsize=10, weight='bold' )
plt.show()


# In[70]:


sns.catplot(x='ORIGIN', kind='count',hue='Flight Status',height=6, data=data, palette='Paired')
plt.title('Delay with respect to origin', fontsize=10, weight='bold' )
plt.show()


# In[71]:


sns.catplot(x='DEST', kind='count',hue='Flight Status',height=6, data=data, palette='Paired')
plt.title('Delay with respect to origin', fontsize=10, weight='bold' )
plt.show()


# In[72]:


sns.boxplot(y='DISTANCE', x='Flight Status', data=data, palette='ch:.39041')
plt.show()


# In[73]:


def test(data):
    for index, row in data.iterrows():
        data.set_value(index, 'FL_DA', (row.TAIL_NUM[1:4]))
        data.set_value(index, 'FL', (row.TAIL_NUM[4:]))
    return data

d=test(data)


# In[74]:


sns.catplot(y='DAY_OF_MONTH', x='FL_DATE',height=4, col='CARRIER', data=d)
plt.title('Month wrt Date', fontsize=10, weight='bold' )
plt.show()


# In[75]:


sns.catplot(y='DAY_OF_MONTH', x='DAY_WEEK',height=4, col='CARRIER', data=d)
plt.title('Month wrt Day', fontsize=10, weight='bold' )
plt.show()


# In[76]:


sns.catplot(y='FL_DA', x='CARRIER',height=4, col='CARRIER', data=d)
plt.title('Carrier wrt Tail_Num', fontsize=10, weight='bold' )
plt.show()


# In[77]:


sns.catplot(y='FL_DA', x='FL',height=4, col='CARRIER', data=d)
plt.show()

