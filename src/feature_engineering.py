import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from statsmodels.discrete.discrete_model import Logit
import matplotlib.pyplot as plt
from datetime import datetime

df_test = pd.read_csv('data/churn_test.csv')
df_test['phone'][df_test['phone'].isnull()] = 'NA'
df_test.groupby('phone')['phone'].count()

df_test['phone'].unique()
def get_data():
    df = pd.read_csv('data/churn_train.csv')
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['churn'] = df['last_trip_date'].apply(lambda x: 1 if x < pd.Timestamp('2014-06-01') else 0)
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    print 'percentage not churn: {}' .format(df[df['churn']==0]['churn'].count()/float(df.shape[0]))
    return df

def get_day(datetime):
    return datetime.today()


#Drop data where avg_dist > 100
#

df = get_data()
df.head()
df.describe().T
df = df.drop(df.ix[df['trips_in_first_30_days']> 70])
df[df['trips_in_first_30_days'] > 70]
df = df[df['trips_in_first_30_days'] < 100]
df = df[df['avg_dist'] < 100]
10.618

df['lowrating'] = 0
df['lowrating'][(df['avg_rating_of_driver']< 2) & (df['avg_rating_of_driver'].isnull()==False)] = 1
df['phone'][df['phone'].isnull()] = 'NA'
df.groupby('phone')['phone'].count()
df.groupby(['churn', 'phone'])['phone'].count()

df[df['trips_in_first_30_days']==0].groupby(['churn', 'phone'])['phone'].count()
df[df['trips_in_first_30_days']>=5].groupby(['churn','phone'])['phone'].count()
df[df['weekday_pct']==0]['weekday_pct'].count()
df[df['weekday_pct']==100]['weekday_pct'].count()
df.head()
df.info()
A = df['weekday_pct']
df[df['avg_dist']>100]
df[df['last_trip_date']==df['signup_date']].count()

df['Weekday'] = 0
df['Weekend'] = 0
df['Weekday'][df['weekday_pct'] < 10] = 1
df['Weekend'][df['weekday_pct'] > 90] = 1


df.head()
df.groupby(['signup_date','churn'])['churn'].count()
df.groupby(['weekday_pct'])['weekday_pct'].count()

plt.plot(df.groupby(['weekday_pct'])['weekday_pct'].count())
plt.show()
#Only ever ridden once

df[(df['avg_surge']!=1) & (df['surge_pct']!=0)]
#Churn
#if avg rating % 1 is 0 or NaN, avg rating of % 1 is 0 or NaN, trips_in_first_30_days <=1
#weekday_pct % 100 == 0, #delete? -> (avg_surge == 1.00 AND surge_pct == 0.0)
df['Custom_cat'] = 0
df['Custom_cat'][(((df['avg_rating_by_driver']<5.0)|(df['avg_rating_by_driver'].isnull())) & (((df['avg_rating_by_driver']%1 == 0)|(df['avg_rating_by_driver'].isnull())) & ((df['avg_rating_of_driver']%1 == 0)|(df['avg_rating_of_driver'].isnull())) & ((df['trips_in_first_30_days']<=1)) & (df['weekday_pct']%100==0) & ((df['avg_surge']==1.00)&(df['surge_pct']==0.00)) ))] = 1
df['Custom_cat'].sum()

df[df['weekday_pct']==0.0]['weekday_pct'].count()
13308+7363
df.head()



#Churn rate by dayofweek
df['last_day'] = df['last_trip_date'].dt.dayofweek
df['signup_day'] = df['signup_date'].dt.dayofweek
last_day_churnrate0 = df['churn'][df['last_day']==0].sum() / float(df['churn'][df['last_day']==0].count())
last_day_churnrate1 = df['churn'][df['last_day']==1].sum() / float(df['churn'][df['last_day']==1].count())
last_day_churnrate2 = df['churn'][df['last_day']==2].sum() / float(df['churn'][df['last_day']==2].count())
last_day_churnrate3 = df['churn'][df['last_day']==3].sum() / float(df['churn'][df['last_day']==3].count())
last_day_churnrate4 = df['churn'][df['last_day']==4].sum() / float(df['churn'][df['last_day']==4].count())
last_day_churnrate5 = df['churn'][df['last_day']==5].sum() / float(df['churn'][df['last_day']==5].count())
last_day_churnrate6 = df['churn'][df['last_day']==6].sum() / float(df['churn'][df['last_day']==6].count())

last_day_churnrate0
last_day_churnrate1
last_day_churnrate2
last_day_churnrate3
last_day_churnrate4
last_day_churnrate5
last_day_churnrate6

signup_day_churnrate0 = df['churn'][df['signup_day']==0].sum() / float(df['churn'][df['signup_day']==0].count())
signup_day_churnrate1 = df['churn'][df['signup_day']==1].sum() / float(df['churn'][df['signup_day']==1].count())
signup_day_churnrate2 = df['churn'][df['signup_day']==2].sum() / float(df['churn'][df['signup_day']==2].count())
signup_day_churnrate3 = df['churn'][df['signup_day']==3].sum() / float(df['churn'][df['signup_day']==3].count())
signup_day_churnrate4 = df['churn'][df['signup_day']==4].sum() / float(df['churn'][df['signup_day']==4].count())
signup_day_churnrate5 = df['churn'][df['signup_day']==5].sum() / float(df['churn'][df['signup_day']==5].count())
signup_day_churnrate6 = df['churn'][df['signup_day']==6].sum() / float(df['churn'][df['signup_day']==6].count())

signup_day_churnrate0
signup_day_churnrate1
signup_day_churnrate2
signup_day_churnrate3
signup_day_churnrate4
signup_day_churnrate5
signup_day_churnrate6
