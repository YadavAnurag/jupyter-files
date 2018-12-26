
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# !wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv


# In[5]:


# read the data
df = pd.read_csv("datasets/fuelConsumptionCo2/fuelConsumptionCo2.csv")

df.head()


# In[7]:


df.describe()


# In[9]:


# extract some features

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head(9)


# In[10]:


# plotting each of these features

viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()


# In[15]:


# Now plotting each of these features vs Emission of CO2 to see how linear the relation is.

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel('CO2EMISSIONS')
plt.show()


# In[20]:


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='purple')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')
plt.show()


# In[21]:


plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='cyan')
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emissions')
plt.show()


# In[22]:


# creating train and test dataset using numpy.random.rand()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# In[24]:


# train data distribution

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='violet')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')
plt.show()


# In[25]:


# modeling - using sklearn package to model the data
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

# the coefficients
print('Coefficients: ', regr.coef_)
print('Intercepts: ', regr.intercept_)


# In[26]:


# plotting the outputs
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='skyblue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')


# In[28]:


# Evaluation

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print('Mean absolute error: %.2f' %np.mean(np.absolute(test_y_hat - test_y)))
print('Residual sum of squares (MSE): %.2f' %np.mean((test_y_hat - test_y)**2))
print('R2-score: %.2f' %r2_score(test_y_hat, test_y))

