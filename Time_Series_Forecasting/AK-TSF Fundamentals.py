#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from pandas import Series
from numpy import log


# In[2]:


os.getcwd()


# In[3]:


os.chdir("D:\Learnbay\Time Series Forecasting\Sundaram Sir TSF")


# In[4]:


os.getcwd()


# In[5]:


dataset =pd.read_csv("D:\\Learnbay\\Time Series Forecasting\\Sundaram Sir TSF\\Datasets\\airline_passengers.csv")


# In[6]:


dataset.head()


# In[7]:


dataset.isnull().sum()


# In[8]:


dataset.info()


# In[9]:


dataset['Month'] = pd.to_datetime(dataset['Month'])


# In[10]:


dataset.info()


# In[11]:


dataset.plot()


# In[12]:


dataset.set_index('Month',inplace =True)


# In[13]:


dataset.plot()


# In[14]:


# Decomposition to check dataset component

from statsmodels.tsa.seasonal import seasonal_decompose

decompose = seasonal_decompose(dataset['Thousands of Passengers'],model = 'additive',period =12)
decompose.plot()
plt.show()


# In[15]:


from statsmodels.tsa.seasonal import seasonal_decompose

decompose = seasonal_decompose(dataset['Thousands of Passengers'],model = 'multiplicative',period =12)
decompose.plot()
plt.show()


# # Check Autocorrelation part -Durbin Watson Test

# In[16]:


import statsmodels.api as sm


# In[17]:


sm.stats.durbin_watson(dataset['Thousands of Passengers'])

#if the value is =2(regression pb)
# less than or greater than 2 is a TSF pb


# # Check Data is Stationary or Non-Stationary

# # Augmented Dickey Fuller  Test(P-value)
# 

# In[18]:


# Augmented Dickey Fuller  Test - check Data Stationary
from statsmodels.tsa.stattools import adfuller


# In[19]:


adfuller(dataset['Thousands of Passengers'])


# In[20]:


def check_adftest(timeseries):
    result = adfuller(timeseries)
    print("Augmented Dickey Fuller Test- To chekc data is Stationary or not")
    labels = ['ADF Test Stats','P-value','No of lags','No of observation']
    
    for i, j in zip(result,labels): #Using zip to combine two lists
        print(j + ":-->" + str(i))
        
    if result[1] <=0.05:    # p-value
        print("Strong evidence against null hypothesis and my time series is Stationary")
        
    else:    
        print("Weak evidence against null hypothesis and my time series is non-stationary")


# In[21]:


check_adftest(dataset['Thousands of Passengers'])


# In[22]:


dataset['1st Diff'] = dataset['Thousands of Passengers']-dataset['Thousands of Passengers'].shift(1)


# In[23]:


dataset.head(11)


# In[24]:


check_adftest(dataset['1st Diff'].dropna())


# In[25]:


dataset['2nd Diff'] = dataset['1st Diff']-dataset['1st Diff'].shift(1)


# In[26]:


dataset.head(10)


# In[27]:


check_adftest(dataset['2nd Diff'].dropna())


# In[28]:


# parameter =p d q (Trend)
# d =2 --> because we have got stationary after 2time diff


# In[29]:


# Calculate Seasonality, we have seen seasonality in the above plot

dataset['Seasonality'] = dataset['Thousands of Passengers'] - dataset['Thousands of Passengers'].shift(12)


# In[30]:


dataset.head(20)
# (1950-01-01) -(1949-01-01) =>115-112 =3


# In[31]:


check_adftest(dataset['Seasonality'].dropna())


# In[32]:


# Trend
### p (Auto Regressive order) :
### d (Integration order): 2
### q (Moving Average order):

# Seasonality
### P (Seasonal Auto Regressive order):
### D (Seasonal Integration order): 1
### Q (Seasonal Moving Average order):

## How to calculate parameter p/P & q/Q
## Ans : 'p' stands for partial autocorrelation and we have to use autoregressive method and 
    # 'q'  stand for autocorrelation and  we have to calculate basis moving avg  


# In[33]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[34]:


# p and P values
# p : Trend
plot_pacf(dataset['2nd Diff'].dropna(),lags =15) #consider stationary trend data
plt.show() # lags as many as we want till we find inside dot in shaded area      


# In[35]:


# Trend
### p : 4 (count wherever the dot came out of shaded area until it finds a complete inside one and the first one is original data, so dont consider)
### d : 2
### q : 

# Seasonality
### P :
### D : 1
### Q :


# In[36]:


# P : Trend
plot_acf(dataset['2nd Diff'].dropna(),lags =15) #consider stationary trend data
plt.show()


# In[37]:


# Trend
### p : 4 (count wherever the dot came out of shaded area until it finds a complete inside one)
### d : 2
### q : 2 (count wherever the dot came out of shaded area until it finds a complete inside one)

# Seasonality
### P :
### D : 1
### Q :


# In[38]:


# P : Seasonality

plot_pacf(dataset['Seasonality'].dropna(),lags =15) #consider stationary trend data
plt.show()


# In[39]:


# Trend
### p : 4 (count wherever the dot came out of shaded area until it finds a complete inside one)
### d : 2
### q : 2 (count wherever the dot came out of shaded area until it finds a complete inside one)

# Seasonality
### P : 2 (count wherever the dot came out of shaded area until it finds a complete inside one)
### D : 1
### Q :


# In[40]:


# Q : Seasonality

plot_acf(dataset['Seasonality'].dropna(),lags =15) #consider stationary trend data
plt.show()


# In[41]:


# Trend
### p : 4 (count wherever the dot came out of shaded area until it finds a complete inside one)
### d : 2
### q : 2 (count wherever the dot came out of shaded area until it finds a complete inside one)

# Seasonality
### P : 2 (count wherever the dot came out of shaded area until it finds a complete inside one)
### D : 1
### Q : 5 (count wherever the dot came out of shaded area until it finds a complete inside one and the first one is original data, so dont consider


# # Building Time Series Forecasting Model - ARIMA

# In[42]:


from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm


# In[43]:


model = sm.tsa.statespace.SARIMAX(dataset["Thousands of Passengers"],order = (4,2,2),
                                 seasonal_order =(2,1,5,12))

result = model.fit()
print(result.summary())


# In[44]:


# order = Trend
# seasonal_order =(2,1,5,12) => 12 is months
# we have seasonality thats why we are using sarima and seasonal order
# Lesser the AIC value better the model
# we have to see  AIC and MODEl parameters from the below values

#Model:SARIMAX(4, 2, 2)x(2, 1, [1, 2, 3, 4, 5], 12) has shown this, it means model is saying we will
    # get better result when we Q= 1,2,3,4 along with 5 to get lower AIC value

# seasonal_order =(2,1,1,12) 
# seasonal_order =(2,1,2,12)  
# seasonal_order =(2,1,3,12)
# seasonal_order =(2,1,4,12)
# seasonal_order =(2,1,5,12)


# In[45]:


model = sm.tsa.statespace.SARIMAX(dataset["Thousands of Passengers"],order = (4,2,2),
                                 seasonal_order =(2,1,4,12))

result = model.fit()
print(result.summary())


# In[46]:


model = sm.tsa.statespace.SARIMAX(dataset["Thousands of Passengers"],order = (4,2,2),
                                 seasonal_order =(2,1,3,12))

result = model.fit()
print(result.summary())


# In[47]:


model = sm.tsa.statespace.SARIMAX(dataset["Thousands of Passengers"],order = (4,2,2),
                                 seasonal_order =(2,1,2,12))

result = model.fit()
print(result.summary())


# In[48]:


model = sm.tsa.statespace.SARIMAX(dataset["Thousands of Passengers"],order = (4,2,2),
                                 seasonal_order =(2,1,1,12))

result = model.fit()
print(result.summary())


# In[49]:


# AIC : 1021.450            :((4, 2, 2)x(2, 1,5,12))
# AIC : 1020.040            :((4, 2, 2)x(2, 1,4,12))
# AIC : 1018.352            :((4, 2, 2)x(2, 1,3,12))
# AIC : 1017.565            :((4, 2, 2)x(2, 1,2,12))
# AIC : 1015.571            :((4, 2, 2)x(2, 1,1,12)) -This is the best one


# In[50]:


len(dataset)


# In[51]:


dataset['Forecast'] = result.predict(start =130, end =144,dynamic =True)


# In[52]:


dataset.tail(20)


# In[53]:


dataset[['Thousands of Passengers','Forecast']].plot()


# In[54]:


# Just checking what happens if we consider partial autocorrelation as correlation and viceversa

# Trend
### p : 2(4)
### d : 2(2)
### q : 4(2)

# Seasonality
### P : 1(2)
### D : 1(1)
### Q : 2(1)



model1 = sm.tsa.statespace.SARIMAX(dataset["Thousands of Passengers"],order = (2,2,4),
                                 seasonal_order =(1,1,2,12))

result1 = model1.fit()
print(result1.summary())



# In[55]:


dataset['Forecast1'] = result1.predict(start =130, end =144,dynamic =True)
dataset[['Thousands of Passengers','Forecast1']].plot()


# In[56]:


# trying with random values

model2 = sm.tsa.statespace.SARIMAX(dataset["Thousands of Passengers"],
                            order = (0,2,1),seasonal_order =(5,2,3,12))
                                 

result2 = model2.fit()
print(result2.summary())

# Unfortunatly ranodm values gave better result, but generally this won't happen


# In[57]:


dataset['Forecast2'] = result2.predict(start =130, end =144,dynamic =True)
dataset[['Thousands of Passengers','Forecast2']].plot()


# In[58]:


dataset.head()


# In[59]:


# removing the other tried columns

dataset = dataset.iloc[:,:-2]
dataset.head()


# # Method 2-(Automation) Apply Itertools

# In[60]:


import itertools


# In[61]:


p = d = q = range(0,2)    #this will try all combination (27)
pdq = list(itertools.product(p,d,q))   # gives trend value
seasonal_pdq = [(x[0],x[1],x[2],12) for x in pdq]      # gives seasonality value

print("Check few paramter combinations are :")
print('{} x {}'.format(pdq[0], seasonal_pdq[0]))
print('{} x {}'.format(pdq[0], seasonal_pdq[1]))
print('{} x {}'.format(pdq[0], seasonal_pdq[2]))
print('{} x {}'.format(pdq[1], seasonal_pdq[0]))
print('{} x {}'.format(pdq[1], seasonal_pdq[1]))
print('{} x {}'.format(pdq[1], seasonal_pdq[2]))
print('{} x {}'.format(pdq[2], seasonal_pdq[0]))
print('{} x {}'.format(pdq[2], seasonal_pdq[1]))
print('{} x {}'.format(pdq[2], seasonal_pdq[2]))


# In[62]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            model = sm.tsa.statespace.SARIMAX(dataset["Thousands of Passengers"],
                            order =param,seasonal_order = param_seasonal,
                        enforce_stationarity=False,enforce_invertibility=False)
            result = model.fit()
            print("ARIMA {} x {} - AIC:{}".format(param, param_seasonal,result.aic))
        except:
            continue
                                          


# In[63]:


# ARIMA (0, 1, 1) x (1, 1, 1, 12) - AIC:920.3192974989249

# why we got d=1 in automation whereas in our manual method we got d=2 because we gave hard code value p<=0.5
     # but in 1st order we got 0.054(difference 0.004) machine has consider the 1st value it neglected the minor difference
    
model3 = sm.tsa.statespace.SARIMAX(dataset["Thousands of Passengers"],
                            order =(0,1,1),seasonal_order = (0,1,1,12),
                        enforce_stationarity=False,enforce_invertibility=False)

result3 = model3.fit()
print(result3.summary())

# AIC is the parameter we need to check for error


# In[64]:


dataset['Forecast3'] = result3.predict(start =130, end =144,dynamic =True)
dataset[['Thousands of Passengers','Forecast3']].plot()


# # Forecast passenger details for 5 years

# In[65]:


dataset.tail(10)


# In[66]:


from pandas.tseries.offsets import DateOffset


# In[67]:


dataset.index[-1] # last month of the existing tabl 


# In[68]:


future_date =[dataset.index[-1] + DateOffset(months =x) for x in range(0,61)]
# Forecasting for 5 years = 60months


# In[69]:


future_date


# In[70]:


future_date_df =pd.DataFrame(index = future_date[1:], columns = dataset.columns)


# In[71]:


future_date_df.tail(60)


# In[72]:


len(dataset)


# In[73]:


len(future_date_df)


# In[74]:


final_date = pd.concat([dataset,future_date_df])


# In[75]:


final_date


# In[76]:


len(final_date)


# In[77]:


# predict future passenger details and visualize it for understanding purpose

final_date['Forecast3'] =result.predict(start =144, end = 204, dynamic =True)
final_date[['Thousands of Passengers','Forecast3']].plot()


# In[78]:


final_date.tail()


# In[79]:


final_date.to_csv("Forecasted Values for next 5 years.csv")


# In[ ]:





# # Method 3-Auto ARIMA Model

# In[80]:


#!pip install pmdarima


# In[81]:


from pmdarima import auto_arima


# In[82]:


mydata = pd.read_csv("D:\\Learnbay\\Time Series Forecasting\\Sundaram Sir TSF\\Datasets\\airline_passengers.csv")


# In[83]:


mydata['Month'] = pd.to_datetime(mydata['Month'])


# In[84]:


mydata.info()


# In[85]:


mydata.set_index('Month',inplace =True)


# In[86]:


mydata.head()


# In[87]:


model = auto_arima(mydata, seasonal =True, m =12)


# In[88]:


print(model.summary())


# In[89]:


# here without doing anything we got AIC value is 1017


# In[91]:


# ************************************** #


# In[95]:


dataset.head(5)


# In[92]:


dataset.tail()


# In[106]:


dataset1 = dataset.copy()


# In[107]:


dataset1 = dataset1[["Thousands of Passengers","Forecast3"]]


# In[108]:


dataset1


# In[109]:


dataset1 = dataset1.iloc[130:,: ]


# In[110]:


dataset1


# In[111]:


actual_value = dataset1['Thousands of Passengers']
predicted_value = dataset1['Forecast3']


# In[113]:


# Calculate MAPE

def calculate_mape(actual,predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted)/actual))*100


# In[114]:


mape = calculate_mape(actual_value, predicted_value)
print("Mean Absolute Percent Error :",mape)


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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




