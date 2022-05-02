#!/usr/bin/env python
# coding: utf-8

# ## Multiple Linear Regression

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[9]:


data = pd.read_csv('Data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 0)


# In[11]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[12]:


y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


# In[13]:


from sklearn.metrics import r2_score
r2_score(y_pred, y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




