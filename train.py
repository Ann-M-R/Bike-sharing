# %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

import seaborn as sns
from matplotlib import pyplot as plt
import pickle
output_file = 'model.bin'
# %%
df = pd.read_csv("london_merged.csv")

# %%
df.head()

# %%
df.columns = df.columns.str.lower()

# %%
df['timestamp'] = pd.to_datetime(df['timestamp'])

# %%
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df['year'] = df['timestamp'].dt.year
df['weekday'] = df['timestamp'].dt.weekday

# %%
df.drop(columns=['timestamp'], inplace=True)

# %% [markdown]
# ### EDA

# %%
df.isnull().sum()

# %%
df.nunique()

# %%
df.head()

# %%
holiday_mapping = {0: "non-holiday", 1: "holiday"}
df['is_holiday'] = df['is_holiday'].map(holiday_mapping)

# %%
weekend_mapping = {0: "weekday", 1: "weekend"}
df['is_weekend'] = df['is_weekend'].map(weekend_mapping)

# %%
season_mapping = {
    0: "spring",
    1: "summer",
    2: "fall",
    3: "winter"
}
df['season'] = df['season'].map(season_mapping)

# %%
df['weather_code'].unique()

# %%
weather_code_mapping = {
    1: "mostly_clear",
    2: "few_clouds",
    3: "broken_clouds",
    4: "cloudy",
    7: "light_rain",
    10: "thunderstorm",
    26: "snowfall"
}

df['weather_code'] = df['weather_code'].map(weather_code_mapping)

# %%
weekday_mapping = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}

df['weekday'] = df['weekday'].map(weekday_mapping)

# %%
df.head()

# %%
df.dtypes

# %%
numerical = ['cnt', 't1', 't2', 'hum', 'wind_speed']

# %%
plt.figure(figsize=(5, 5))
sns.histplot(df['cnt'], bins=50, kde=True)


# %%
categorical = ['weather_code', 'is_holiday', 'is_weekend', 'season', 'month', 'weekday', 'year', 'hour']
numerical = ['t1', 't2', 'hum', 'wind_speed', 'day']

# %%
from sklearn.metrics import mutual_info_score
def mutual_info_diag_score(series):
    return mutual_info_score(series, df.cnt)

mi = df[categorical].apply(mutual_info_diag_score)
mi

mi.sort_values(ascending = False)

# %%
df[numerical].corrwith(df.cnt)

# %%
correlation_matrix = df[numerical].corr()
# %% [markdown]
# Splitting the datset

# %%
from sklearn.model_selection import train_test_split   

# %%
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
len(df_full_train), len(df_test)

# %%
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
len(df_train), len(df_val)

# %%
df_train = df_train.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)

# %%
df_train.head()

# %%
y_train = df_train.cnt.values
y_test = df_test.cnt.values
y_val = df_val.cnt.values
# %%
del df_train['cnt']
del df_test['cnt']
del df_val['cnt']

# %% [markdown]
# ### Simple Regression
# %%
train_dicts = df_train[categorical + numerical].to_dict(orient='records')
val_dicts = df_val[categorical + numerical].to_dict(orient='records')

# %%
dv = DictVectorizer(sparse=False)

# %%
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

# %%
X_train.shape

# %%
dv.get_feature_names_out()
# %% [markdown]
# ### Gradient Boosting

# %%
params = {
    "eta": 0.1,
    "n_estimators": 125,
    "max_depth": 10,
    "min_child_weight": 5,
    "objective": "reg:squarederror",
    "eval_metric": "rmse", 
    "random_state": 1,
}

xgb = XGBRegressor(**params)

xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

y_pred = xgb.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5

print("training final model")

y_full_train = df_full_train['cnt']
del df_full_train['cnt']
dicts_full_train = df_full_train.to_dict(orient='records')
 
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)
 
dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

y_pred_test = xgb.predict(X_test)

mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = mse_test ** 0.5

print(f"Final RMSE on Test Data: {rmse_test}")

with open (output_file, 'wb') as f_out:
    pickle.dump((dv,xgb), f_out) 


print(f'model saved to {output_file}')



