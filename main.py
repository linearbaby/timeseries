#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller


# In[3]:


data = pd.read_excel("Данные.xlsx")
data = data.rename(columns={"дата": "date", "направление": "dir", "выход": "out"}).set_index("date")
data = data.iloc[::-1]
data.head()

# In[4]:


result_df = pd.read_excel("Данные.xlsx", sheet_name=1, index_col=0)
result_df.head()

# In[5]:


plt.figure(figsize=(10, 6))
plt.plot(data.index, data["out"], label='Simulated Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# In[6]:


adfuller(data["out"], regression="ctt")

# Можно заметить, что это ряд типа DSP, который нужно остационаривать процедурой взятия разностей.

# In[7]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(10, 6))
plot_acf(data.out.diff(1).dropna(), lags=20)
plt.show()

# Plot Partial Autocorrelation (PACF)
plt.figure(figsize=(10, 6))
plot_pacf(data.out.diff(1).dropna(), lags=20, method='ywm')
plt.show()

# Можно заметить, что моделирование первых разностей напоминает белый шум.

# ### EDA, составление датасета

# In[8]:


train_size = int(0.8 * data.shape[0])
train, test = data["out"].iloc[:train_size], data["out"].iloc[train_size:]

# In[9]:


from statsmodels.tsa.arima.model import ARIMA

# Fit AR(1) model
model_ar1 = ARIMA(train, order=(4, 2, 1))
model_ar1_fit = model_ar1.fit()

# In[10]:


arima_res = model_ar1_fit.forecast(steps=data.shape[0] - train_size)
arima_res

# In[11]:


plt.figure(figsize=(10, 6))
plt.plot(test.index, test.values, label='initial Time Series')
arima_res = model_ar1_fit.forecast(steps=data.shape[0] - train_size)
plt.plot(test.index, arima_res.values, label='Simulated Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# In[12]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(arima_res.values, test)

# ### ML model

# In[13]:


from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.linear_model import LinearRegression

# In[14]:


def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_macd(series, short_period, long_period, signal_period):
    ema_short = calculate_ema(series, short_period)
    ema_long = calculate_ema(series, long_period)
    macd = ema_short - ema_long
    signal_line = calculate_ema(macd, signal_period)
    return macd, signal_line

def calculate_rsi(series, period):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def reconstruct_value(prev_value, logs, type_="diff"):
    if type_ == "diff":
        result = np.exp(logs) + prev_value
    else:
        result = np.exp(logs) * prev_value
    return result

def autoregressive(
    model, 
    out_start, 
    steps,
    categorical,
    numerical,
):
    results = list(out_start)
    for step in range(steps):
        cur_data = pd.DataFrame({"out": results})
        preprocessed = generate_features(cur_data)
        # print(preprocessed)
        X_test = preprocessed[categorical + numerical]

        predictions = model.predict(X_test.iloc[-1:])
        predictions = np.exp(predictions[0]) * results[-1]
        results.append(predictions)
    
    return results


#### minimum for feature generation
minima_length = 30
def generate_features(data):
    # Feature engineering
    data = data.copy()
    for lag in range(1, 30, 5):
        data[f'log_lag0_lag{lag}'] = np.log(data['out'] / data['out'].shift(lag))
    for ema_range in [5, 10, 20, 30]:
        data[f'ema_{ema_range}'] = calculate_ema(data['out'], ema_range)
    macd, signal = calculate_macd(data['out'], 12, 26, 9)
    data['macd'] = macd
    data['macd_signal'] = signal
    for rsi_range in [5, 10, 20, 30]:
        data[f'rsi_{rsi_range}'] = calculate_rsi(data['out'], rsi_range)
    data["prev_value"] = data["out"].shift(1)

    # Dropping rows with NaNs caused by the indicators
    return data.dropna()

def generate_target(data, step=1):
    data = data.copy()
    data["target"] = np.log(data["out"].shift(-step) / data["out"])
    return data.dropna()

preprocessed_data = generate_features(data)
preprocessed_data = generate_target(preprocessed_data, step=1)

# Splitting the dataset again, now with new features
### данные вверх ногами, поэтому такая путаница
split_point = int(len(preprocessed_data) * 0.8)
train_data = preprocessed_data[:split_point]
test_data = preprocessed_data[split_point:]

# Defining new features and target
categorical = ["dir"]
categorical = []
numerical = preprocessed_data.columns.drop(["dir", "out", "target", "prev_value"]).tolist()
X_train, y_train = train_data[categorical + numerical], train_data['target']
X_test, y_test = test_data[categorical + numerical], test_data['target']

# Initialize and fit the CatBoostRegressor
model = CatBoostRegressor(verbose=False)
model.fit(
    X_train, 
    y_train, 
    early_stopping_rounds=100,
    cat_features=categorical, 
    eval_set=(X_test, y_test), 
    verbose=True
)
# model = LinearRegression()
# model.fit(X_train, y_train)

preprocessed_data = generate_features(data)
preprocessed_data = generate_target(preprocessed_data, step=1)

categorical = ["dir"]
categorical = []
numerical = preprocessed_data.columns.drop(["dir", "out", "target", "prev_value"]).tolist()
preprocessed_data = preprocessed_data[categorical + numerical]

# In[17]:


result = autoregressive(
    model, 
    data["out"].iloc[-minima_length:], 
    steps = len(result_df),
    categorical=categorical,
    numerical=numerical,
)
result = result[-len(result_df):]

# In[18]:


result_df["выход"] = result

# ### Теперь предскажем направление

# In[21]:


#### minimum for feature generation
minima_length = 30
def generate_features(data):
    # Feature engineering
    data = data.copy()
    for lag in range(1, 30, 5):
        data[f'log_lag0_lag{lag}'] = np.log(data['out'] / data['out'].shift(lag))
    for ema_range in [5, 10, 20, 30]:
        data[f'ema_{ema_range}'] = calculate_ema(data['out'], ema_range)
    macd, signal = calculate_macd(data['out'], 12, 26, 9)
    data['macd'] = macd
    data['macd_signal'] = signal
    for rsi_range in [5, 10, 20, 30]:
        data[f'rsi_{rsi_range}'] = calculate_rsi(data['out'], rsi_range)
    data["prev_value"] = data["out"].shift(1)
    for lag in range(1, 30, 3):
        data[f'dir_lag{lag}'] = data['dir'].shift(lag)

    # Dropping rows with NaNs caused by the indicators
    return data.dropna()

def generate_target(data, step=1):
    data = data.copy()
    data["target_class"] = data["dir"].shift()
    return data.dropna()

def autoregressive(
    model, 
    model_class,
    out_start, 
    steps,
    categorical,
    numerical,
):
    results = list(out_start["out"])
    results_class = list(out_start["dir"])
    for step in range(steps):
        cur_data = pd.DataFrame({"out": results, "dir": results_class})
        preprocessed = generate_features(cur_data)
        # print(preprocessed)
        X_test = preprocessed[categorical + numerical]
        # print(X_test)

        predictions = model.predict(X_test.iloc[-1:].drop(columns=X_test.filter(like="dir").columns))
        predictions_class = model_class.predict(X_test.iloc[-1:])
        predictions = np.exp(predictions[0]) * results[-1]
        results.append(predictions)
        results_class.append(predictions_class[0])
    
    return results_class

preprocessed_data = generate_features(data)
preprocessed_data = generate_target(preprocessed_data, step=1)

split_point = int(len(preprocessed_data) * 0.8)
train_data = preprocessed_data[:split_point]
test_data = preprocessed_data[split_point:]

# Defining new features and target
categorical = preprocessed_data.filter(like="dir").columns.tolist()
numerical = preprocessed_data.columns.drop(["out", "target_class", "prev_value"] + categorical).tolist()
X_train, y_train = train_data[categorical + numerical], train_data['target_class']
X_test, y_test = test_data[categorical + numerical], test_data['target_class']

# Initialize and fit the CatBoostRegressor
model_cat = CatBoostClassifier(verbose=False)
model_cat.fit(
    X_train, 
    y_train, 
    early_stopping_rounds=100,
    cat_features=categorical, 
    eval_set=(X_test, y_test), 
    verbose=True
)
# model = LinearRegression()
# model.fit(X_train, y_train)

# Predicting and evaluating again
predictions = model_cat.predict(X_test)

label_to_idx = {
    "л": 1,
    "ш": 0,
}
mse = f1_score(
    y_test.map(label_to_idx),
    list(map(lambda x: label_to_idx[x], predictions))
)
print(f'Mean Squared Error with Feature Engineering: {mse}')

# In[22]:


result = autoregressive(
    model, 
    model_cat,
    data.iloc[-minima_length:], 
    steps = len(result_df),
    categorical=categorical,
    numerical=numerical,
)
result = result[-len(result_df):]
result

# In[23]:


result_df["направление"] = result
result_df["направление"] = result_df["направление"].map(label_to_idx)

# In[24]:


import json
data1 = result_df["выход"].tolist()
data2 = result_df["направление"].tolist()
with open('forecast_value.json', 'w') as file:
    json.dump(data1, file)
with open('forecast_class.json', 'w') as file:
    json.dump(data2, file)
