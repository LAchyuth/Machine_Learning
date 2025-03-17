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
warnings.filterwarnings('ignore')

from datetime import datetime
from pandas import Series
from numpy import log

import plotly.offline as py


# In[2]:


os.getcwd()


# In[3]:


os.chdir("D:\Learnbay\Time Series Forecasting\Sundaram Sir TSF")


# In[4]:


os.getcwd()


# In[5]:


#  Steps to remember before doing the TSF pb
# 1. Every dat should be numeric value
# 2 Data type is correct or not- datetiem column and target variable should be numeric
# 3 Ther should not be an missing data, if yes, first impute and then try to solve the tsf
# 4 Data should be in sequential order (TS is alway in an ascending order)
# 5 Whether data has trend, seasonality error or abrupt changes or something in x changes


# In[6]:


df = pd.read_csv("D:\\Learnbay\\Time Series Forecasting\\Sundaram Sir TSF\\Datasets\\TCS.NS.csv")


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


print("No of observation:",df.shape[0])
print("No of variables:",df.shape[1])


# In[10]:


df['Date'] = pd.to_datetime(df['Date'])


# In[11]:


df.info()


# In[12]:


df.isnull().sum()


# In[13]:


import statsmodels.api as sm
sm.stats.durbin_watson(df['Volume'])


# In[14]:


print(sm.stats.durbin_watson(df['Open']))
print(sm.stats.durbin_watson(df['Close']))
print(sm.stats.durbin_watson(df['High']))
print(sm.stats.durbin_watson(df['Low']))


# In[15]:


# all the above values are either lesser or greater than 2, so it is a TSF pb


# ### Series has to be read as a daily series with Mon -Fri as weekdays, hence frequency is defined as business day, else if you define frequency as 365, it would assume that exchange was working for all 365 days

# In[16]:


from pandas.tseries.offsets import BDay # BDay -business day(mon -fri)


# In[17]:


df.shape


# In[18]:


df.head()


# In[19]:


df_date = df[["Date","Volume"]]
# instead of doing with all variables,we are predicting Volume only (Target variable)


# In[20]:


df_date.head()


# In[21]:


#df.set_index('Date',inplace = True)


# In[22]:


df_date.head()


# In[23]:


df_date.tail()


# In[24]:


date1 = pd.date_range(start ='3/14/2019', end ='3/13/2024',freq = BDay())
date1

# in the df, we have total 1235 days
# in the date1, we have total 1305 days ((365*5)-104*5) =1305
# In[25]:


from pandas.tseries.holiday import USFederalHolidayCalendar  # just working with us calender,we can do with indian
from pandas.tseries.offsets import CustomBusinessDay  # to customize ac to business
us_bd = CustomBusinessDay(calendar =USFederalHolidayCalendar())
us_bd


# In[26]:


date1 = pd.date_range(start ='3/14/2019', end ='3/13/2024',freq =us_bd)
date1

# among 1305, there will be holdays for festival, after removing all we got 1252
# In[27]:


df_final = pd.read_csv("D:\\Learnbay\\Time Series Forecasting\\Sundaram Sir TSF\\Datasets\\TCS.NS.csv",
                       parse_dates = True, squeeze =True, index_col =0)

# parse_dates = True => (dates are converting into date format even if we hadn't done in the above this would have considered)
# squeeze =True => True  Pandas will attempt to load that data into a Series instead of a DataFrame
# index_col =0  => this will consider the first coulmn as index column(date), hence no need to do set_index = date


# In[28]:


df_final.head()


# In[29]:


df_final.info()


# In[30]:


plt.figure(figsize =(15,8))
df_final.plot()
plt.grid()


# ### Plot the boxplot of the "Open" variable with respect to the different years

# In[31]:


sns.boxplot(x =df_final.index.year, y = df_final['Open'])
plt.grid()


# ### Plot the boxplot of the "Open" variable with respect to the different Months

# In[32]:


plt.figure(figsize =(20,8))
sns.boxplot(x =df_final.index.month_name(), y = df_final['Open'])
plt.grid()


# In[33]:


# best month to buy shares is sep & oct


# In[34]:


# Decomposition
plt.figure(figsize =(20,8))
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df_final['Volume'],period =5)
#period =5 because we can in the above plot we can see a pattern for 5 month
decomposition.plot()
plt.show()


# In[35]:


trend = decomposition.trend
seasonality = decomposition.seasonal
resid = decomposition.resid

print("Trend", "\n", trend.head(12),'\n')
print("Seasonal", "\n", seasonality.head(12),'\n')
print("Trend", "\n", resid.head(12),'\n')


# In[36]:


df_final.head()


# In[37]:


df_final['year'] = df_final.index.year
df_final['month'] = df_final.index.month
df_final['Days'] = df_final.index.day
df_final.head()

# Here we want date as column not an index,
# In[38]:


df = pd.read_csv("D:\\Learnbay\\Time Series Forecasting\\Sundaram Sir TSF\\Datasets\\TCS.NS.csv")


# In[39]:


df.head()


# In[40]:


df.info()


# In[41]:


df['Time_stamp'] = pd.to_datetime(df['Date'])


# In[42]:


df.head()


# In[43]:


df.info()


# In[44]:


df_final_model = df.set_index('Time_stamp')
df_final_model.head()


# In[45]:


df_final_model['year'] = df_final_model.index.year
df_final_model['month'] = df_final_model.index.month
df_final_model['Days'] = df_final_model.index.day
df_final_model.head()


# In[46]:


df_final_model.tail()


# In[47]:


df_final_model.shape


# In[48]:


df_final_model.info()


# In[49]:


train = df_final_model[pd.to_datetime(df_final_model['Date']) < pd.to_datetime('2023-06-01')]
train.shape
# pd.to_datetime used here because the date is not in datetime format


# In[50]:


test = df_final_model[pd.to_datetime(df_final_model['Date']) >= pd.to_datetime('2023-06-01')]
test.shape
# pd.to_datetime used here because the date is not in datetime format


# In[51]:


train.tail()


# In[52]:


test.head()


# In[53]:


test.tail()


# In[54]:


train_final = train[['Volume']]
test_final = test[['Volume']]


# In[55]:


train_final.head()


# In[56]:


test_final.tail()


# # We will check whether the data is stationary or not using Statistical /Hypothesis Test(p-value <0.05)

# In[57]:


# Method -2
# Augmented Dickey Fuller Test
from statsmodels.tsa.stattools import adfuller

# there are two method available- hypothesis testing and Rolling Stats

def test_stationariy(timeseries):
    rolmean = timeseries.rolling(window =7).mean()  # for weekly data as per our requirement
    rolstd = timeseries.rolling(window =7).std()
    
    # plot rolling stats
    orig = plt.plot(timeseries, color ='blue', label ="Original")
    mean = plt.plot(rolmean, color ='red', label ='rolling mean')
    std = plt.plot(rolstd, color ='black', label ='rolling std')
    plt.legend(loc ='best')
    plt.title("Rolling Mean and Standard Deviation - Stationary or Not!!")
    plt.show(block =False)
    
    # perform Dickey Fuller Test
    
    print("Results of Dickey Fuller Test")
    dftest = adfuller(timeseries, autolag ="AIC")  #df = dickey fulller
    dfoutput = pd.Series(dftest[0:4], index = ['ADF Test stats', 'P-value','# Lags',"# Observations"])
    for key, value in dftest[4].items():
        dfoutput['Critical values (%s)' %key] =value
    print(dfoutput, '\n')    


# In[58]:


test_stationariy(train_final['Volume'])


# In[59]:


# Method -1

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


# In[60]:


check_adftest(train_final['Volume'])


# # Tseries is Stationary at 5% alpha values as per Dickey Fuller Test

# In[61]:


import itertools
p = q = range(0,3)  # we dont know this
d = range(0,1)  # we already know that data become stationary in the 1st iteration itself(so '0')
pdq = list(itertools.product(p,d,q))  # trend

model_pdq = [(x[0],x[1],x[2],5) for x in list(itertools.product(p,d,q))]   # seasonality[P,D,Q]
# here 5 -because we have seen pattern for 5 months in the boxplot
print("Example of parameter combinations for Model....")
print("Model : {}{}".format(pdq[1], model_pdq[1]))
print("Model : {}{}".format(pdq[1], model_pdq[2]))
print("Model : {}{}".format(pdq[1], model_pdq[0]))
print("Model : {}{}".format(pdq[0], model_pdq[1]))
print("Model : {}{}".format(pdq[2], model_pdq[1]))


# # Building ARIMA Model

# In[62]:


# Creating an empty DataFrame with column names
# from the decomposition we can see only trend with no seasonality, but wetry all models

dfobj = pd.DataFrame(columns =['Param', 'AIC'])
dfobj


# In[63]:


from statsmodels.tsa.arima.model import ARIMA

for param in pdq:
    mod = ARIMA(train_final['Volume'], order = param)
    results_ARIMA = mod.fit()
    print("ARIMA{} - AIC:{}".format(param, results_ARIMA.aic))
    dfobj = dfobj.append({'param':param, 'AIC' :results_ARIMA.aic}, ignore_index = True)


# In[64]:


dfobj.sort_values(by = ['AIC'])


# In[65]:


# ARIMA (1,0,2) has the lowest AIC

model = ARIMA(train_final['Volume'], order =(1,0,2),enforce_stationarity=False,
                 enforce_invertibility=False)
results_ARIMA = model.fit()
print(results_ARIMA.summary())


# In[66]:


results_ARIMA.plot_diagnostics(figsize =(16,8))
plt.show()


# In[67]:


final_hat_avg = test_final.copy()
pred_ARIMA =results_ARIMA.forecast(steps = len(test_final))
pred_ARIMA


# In[68]:


len(test_final) # no of days to be predicted 


# In[108]:


from math import sqrt
from sklearn.metrics import mean_squared_error
rmse1 = sqrt(mean_squared_error(test_final.Volume,pred_ARIMA, squared = False))
print(rmse1)


# In[70]:


# Calculate MAPE

def calculate_mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual))*100


# In[109]:


mape1 = calculate_mape(test_final.Volume, pred_ARIMA)
print("MAPE1 :", mape1)


# In[110]:


resultDF = pd.DataFrame({'Test RMSE1' : rmse1}, index = ['ARIMA(1,0,2)'])
resultDF


# In[73]:


plt.plot(train_final, label ='Training Data')
plt.plot(test_final, label = 'Test Data')
plt.plot(test_final.index, pred_ARIMA, label ='Predicted Data - ARIMA Model')
plt.legend(loc ='best')
plt.grid()

# The model is very pathetic, it has to predict like orange color, but it has preidcted average, which is completely wrong
# # Build SARIMA Model

# In[92]:


dfobj2 = pd.DataFrame(columns =['Param','Seasonal','AIC'])
dfobj2


# In[93]:


import statsmodels.api as sm

for param in pdq:
    for param_seasonal in model_pdq:
        model = sm.tsa.statespace.SARIMAX(train_final['Volume'], order = param,
        seasonal_order = param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
    
        # here we have taken sarimax instead of sarima because we dont have SARIMA model alone but we are not considering,
              # sarimax parameters like high, low, open etc other external factors
        results_SARIMA = model.fit()
        print('SARIMA{}x{}5 -AIC:{}'.format(param, param_seasonal, results_SARIMA.aic)) # here 5 some kind of format, no specific meaning
        dfobj2 = dfobj2.append({'Param':param,"Seasonal":param_seasonal,'AIC':results_SARIMA.aic}, ignore_index =True)
        


# In[94]:


dfobj2.sort_values(by =['AIC'])


# In[96]:


# (2, 0, 2)	(2, 0, 2, 5)	32052.602030
model = sm.tsa.statespace.SARIMAX(train_final['Volume'], order = (2,0,2),
        seasonal_order = (2,0,2,5),enforce_stationarity=False,enforce_invertibility=False)

results = model.fit()
print(results.summary())


# In[98]:


results.plot_diagnostics(figsize = (16,8))
plt.show()


# In[99]:


final_hat_avg = test_final.copy()
pred_SARIMA = results.get_forecast(steps = len(test_final))
pred_SARIMA.predicted_mean


# In[111]:


from math import sqrt
from sklearn.metrics import mean_squared_error
rmse2 = sqrt(mean_squared_error(test_final.Volume,pred_SARIMA.predicted_mean, squared = False))
print(rmse2)

# The above rmse clearly states that SARIMA is better than ARIMA as it gave less RMSE value
# In[112]:


mape2 = calculate_mape(test_final.Volume, pred_SARIMA.predicted_mean)
print("MAPE2 :", mape2)

# still the model is giving 33% error, which is not good, so we can improve the model by adding some external factors
# here we are finding volume and it has relation with other parameters like high, low,close,open & adj close
# In the above SARIMAX we used only volume as we dont have SARIMA funtion
# In[114]:


resultDF1 = pd.DataFrame({'Test MAPE' : mape1}, index = ['ARIMA(1,0,2)'])
resultDF2 = pd.DataFrame({'Test MAPE' : mape2}, index = ['SARIMA(2,0,2),(2,0,2,5)'])
resultDF_final = pd.concat([resultDF1,resultDF2])
resultDF_final


# In[116]:


plt.plot(train_final, label ='Training Data')
plt.plot(test_final, label = 'Test Data')
plt.plot(test_final.index, pred_ARIMA, label ='Predicted Data - ARIMA Model')
plt.plot(test_final.index,pred_SARIMA.predicted_mean, label ='Predicted Data - SARIMA Model')
plt.legend(loc ='best')
plt.grid()


# # Building SARIMAX Model -Including EXternal Factor

# In[118]:


df_final.head(2)


# In[119]:


## Let's create exogenious variable
## Open, High, Low, Close, Adj Close

ex_train = train[['Open','High','Low','Close','Adj Close']]
ex_test = test[['Open','High','Low','Close','Adj Close']]
ex_train.head()


# In[121]:


dfobj3 = pd.DataFrame(columns = ['Param','Seasonal','AIC'])
dfobj3


# In[134]:


# Let's first use SARIMAX with exogenous variable( external variable)
# we can select all variables at once  and compare mape with individual one's
# else we can check one by one and eliminate the one which is not required

for param in pdq:
    for param_seasonal in model_pdq:
        model = sm.tsa.statespace.SARIMAX(train_final['Volume'],exog =ex_train,
                     order = param,seasonal_order = param_seasonal,
                     enforce_stationarity=False,enforce_invertibility=False)
        results_SARIMAX = model.fit()
        print('SARIMAX{}{} - AIC:{}'.format(param,param_seasonal,results_SARIMAX.aic))
        dfobj3 = dfobj3.append({'Param':param,'Seasonal':param_seasonal,
                'AIC':results_SARIMAX},ignore_index =True)
        
        


# In[135]:


dfobj3.sort_values(by=['AIC'])


# In[136]:


model = sm.tsa.statespace.SARIMAX(train_final['Volume'],exog =ex_train,
                     order = (2,0,2),seasonal_order = (0,0,2,5),
                     enforce_stationarity=False,enforce_invertibility=False)
results = model.fit()
print(results.summary())


# In[137]:


pred_SARIMAX = results.get_forecast(steps = len(test),exog = ex_test)
pred_SARIMAX.predicted_mean


# In[138]:


rmse3 = sqrt(mean_squared_error(test_final.Volume,pred_SARIMAX.predicted_mean, squared = False))
print(rmse3)


# In[139]:


mape3 = calculate_mape(test_final.Volume, pred_SARIMAX.predicted_mean)
print("MAPE1 :", mape3)


# In[ ]:


# Here actually the model is not working properly with the given external factors, so we have to do
      # one by one and then we have to get least  mape


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





# In[79]:


#complaints@eci.guv.in - for complaints


# In[80]:


# 18004251950-election commission


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




