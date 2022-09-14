#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


users = pd.read_csv("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user")


# In[3]:


users= users['user_id|age|gender|occupation|zip_code'].str.split('|', expand=True)


# In[4]:


users


# In[5]:


users_pd = pd.DataFrame(users)


# In[6]:


print(users_pd)


# In[7]:


users_pd.rename(columns = {0:'user_id'}, inplace = True)


# In[8]:


users_pd


# In[9]:


users_pd.rename(columns = {1:'age'}, inplace = True)


# In[10]:


users_pd.rename(columns = {2:'gender'}, inplace = True)


# In[11]:


users_pd.rename(columns = {3:'occupation'}, inplace = True)


# In[12]:


users_pd.rename(columns = {4:'zip_code'}, inplace = True)


# In[13]:


users_pd


# In[14]:


users.set_index('user_id')


# In[15]:


users.head(25)


# In[16]:


users.tail(10)


# In[17]:


users.shape


# In[18]:


len(users.columns)


# In[19]:


users.dtypes


# In[20]:


print(users['occupation'].to_string(index=False)) 


# In[21]:


duplicateRows = users[users.duplicated(['occupation'])]
duplicateRows


# In[22]:


len(users.occupation)


# In[23]:


len(users['occupation'].unique())


# In[39]:


users.info()


# In[25]:


users.occupation.describe()


# In[42]:





# In[43]:


users["age"] = users["age"].astype(str).astype(int)


# In[44]:


user_mean = users["age"].mean()
user_mean


# In[46]:


users_new = users.explode('age')
users_new['age'].value_counts()


# In[53]:


users['age'].value_counts().idxmin()


# In[ ]:




