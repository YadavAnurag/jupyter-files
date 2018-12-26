
# coding: utf-8

# In[13]:


#importing the packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


# reading the data
df = pd.read_csv('datasets/fuelConsumptionCO2/fuelConsumptionCO2.csv')
df.head()


# In[15]:


# extract some featuers
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head(9)


# In[16]:


# plotting emission value with respect to enginesize
plt.scatter(df.ENGINESIZE, df.CO2EMISSIONS, color='cyan')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')
plt.show()


# In[17]:


# craeting train and test dataset
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# In[10]:


# train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='violet')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')
plt.show()


# In[22]:


# multiple regression model
from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x,y)

# the coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)


# In[24]:


# prediction
y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print('Residual sum of squares: %.2f' %np.mean((y_hat - y)**2))

# as we know variance score: 1 is perfect prediction
print('Variance score: %.2f' %regr.score(x,y))


# In[26]:


regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x, y)
print('Coefficients: ', regr.coef_)

y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print('Residual sum of squares: %.2f' %np.mean((y_hat - y)**2))
print('Variance score: %.2f' %regr.score(x,y))

