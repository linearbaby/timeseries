#!/usr/bin/env python
# coding: utf-8

# In[185]:


import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller


# In[186]:


data = pd.read_excel("Данные.xlsx")
data = data.rename(columns={"дата": "date", "направление": "dir", "выход": "out"}).set_index("date")
data = data.iloc[::-1]
data.head()

# In[187]:


result_df = pd.read_excel("Данные.xlsx", sheet_name=1, index_col=0)
result_df.head()

# In[57]:


plt.figure(figsize=(10, 6))
plt.plot(data.index, data["out"], label='Simulated Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# In[5]:


# Можно заметить, что это ряд типа DSP, который нужно остационаривать процедурой взятия разностей.

# In[6]:

# ### Статистическая модель

# In[118]:


data.index = data.index.to_period('d')

# In[119]:


train_size = int(0.8 * data.shape[0])
train, test = data["out"].iloc[:train_size], data["out"].iloc[train_size:]

# In[180]:


import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

def stl_arima_grid_search(data, test, p_range, d_range, q_range, stl_range, trend_range, seasonal_period, criterion='aic'):
    """
    Perform a grid search to find the best ARIMA model parameters after STL decomposition,
    based on the AIC or BIC criterion.

    Parameters:
    - data: The time series data.
    - p_range, d_range, q_range: Ranges of parameters (p, d, q) for the ARIMA component.
    - seasonal_period: The seasonal period for STL decomposition.
    - criterion: 'aic' for Akaike Information Criterion or 'bic' for Bayesian Information Criterion.

    Returns:
    - A tuple containing the best parameters and the corresponding AIC or BIC value.
    """
    
    best_score, best_cfg = float("inf"), None
    
    for p, d, q, period, trend in itertools.product(p_range, d_range, q_range, stl_range, trend_range):
        try:
            stl = STLForecast(
                train,
                ARIMA,
                model_kwargs=dict(order=(p, d, q), trend=None),
                period=period,
                robust=False,
                seasonal=13, 
                trend=trend
            )

            result = stl.fit()
            forecasted = result.get_prediction(start=len(data), end=len(data) + len(test) - 1).predicted_mean.to_list()
            
            if criterion == 'aic':
                score = result.model_result.aic
            if criterion == 'mse':
                score = mean_squared_error(test, forecasted)
            else: # 'bic'
                score = result.model_result.bic
                
            if score < best_score:
                best_score, best_cfg = score, {"order": (p, d, q), "period": period, "trend": trend}
                print('ARIMA%s %s=%.3f' % ((p, d, q), criterion.upper(), score))
        except:
            continue

    print(f'Best ARIMA{best_cfg} {criterion.upper()}={best_score:.3f}')
    return best_cfg, best_score


# Define the range of parameters to search
p_range = range(0, 4)
d_range = range(0, 5)
q_range = range(0, 4)
stl_range = [92]
trend_range = [int(365 / 2), 365]
seasonal_period = 7  # Assuming a yearly seasonality

# Find the best model based on AIC
best_params, best_score = stl_arima_grid_search(train, test, p_range, d_range, q_range, stl_range, trend_range, seasonal_period, 'mse')


# In[196]:


# Decompose the train data
stl = STLForecast(
        data["out"],
        ARIMA,
        model_kwargs=dict(order=best_params["order"], trend=None),
        period=best_params["period"],
        robust=False,
        seasonal=13, 
        trend=best_params["trend"])

result = stl.fit()
forecasted = result.get_prediction(start=len(data["out"]), end=len(data["out"]) + len(result_df) - 1).predicted_mean.to_list()

print(result.model_result.aic)

# Decompose the train data
stl = STLForecast(
        data["out"],
        ARIMA,
        model_kwargs=dict(order=best_params["order"], trend=None),
        period=best_params["period"],
        robust=False,
        seasonal=13, 
        trend=best_params["trend"])

result = stl.fit()
forecasted = result.get_prediction(start=len(data["out"]), end=len(data["out"]) + len(result_df) - 1).predicted_mean.to_list()

result_df["выход"] = forecasted

# ### ML model


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

# Predicting and evaluating again
predictions = model.predict(X_test)
mse = mean_squared_error(
    reconstruct_value(test_data["out"], y_test, type_=""), 
    reconstruct_value(test_data["out"], predictions, type_="")
)
print(f'Mean Squared Error with Feature Engineering: {mse}')


# ### вывод результатов

# ### Теперь предскажем направление

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

#### minimum for feature generation
minima_length = 30
def generate_features(data):
    # Feature engineering
    data = data.copy()
    for lag in range(1, 30, 1):
        data[f'log_lag0_lag{lag}'] = np.log(data['out'] / data['out'].shift(lag))
    for ema_range in [5, 10, 20, 30]:
        data[f'ema_{ema_range}'] = calculate_ema(data['out'], ema_range)
    macd, signal = calculate_macd(data['out'], 12, 26, 9)
    data['macd'] = macd
    data['macd_signal'] = signal
    for rsi_range in [5, 10, 20, 30]:
        data[f'rsi_{rsi_range}'] = calculate_rsi(data['out'], rsi_range)
    for lag in range(1, 30, 1):
        data[f'dir_lag{lag}'] = data['dir'].shift(lag)

    # Dropping rows with NaNs caused by the indicators
    return data.dropna()

def generate_target(data, step=1):
    data = data.copy()
    data["target_class"] = data["dir"].shift(-1)
    return data.dropna()

def autoregressive(
    model_class,
    out_start, 
    price_prediction,
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

        predictions_class = model_class.predict(X_test.iloc[-1:])
        results.append(price_prediction[step])
        results_class.append(predictions_class[0])
    return results_class

preprocessed_data = generate_features(data)
preprocessed_data = generate_target(preprocessed_data, step=1)

split_point = int(len(preprocessed_data) * 0.8)
train_data = preprocessed_data[:split_point]
test_data = preprocessed_data[split_point:]

# Defining new features and target
categorical = preprocessed_data.filter(like="dir").columns.tolist()
numerical = preprocessed_data.columns.drop(["out", "target_class"] + categorical).tolist()
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
print(f'F1 score with Feature Engineering: {mse}')

# In[251]:


result = autoregressive(
    model_cat,
    data.iloc[-minima_length:], 
    result_df["выход"].tolist(),
    steps = len(result_df),
    categorical=categorical,
    numerical=numerical,
)
result = result[-len(result_df):]

# In[252]:


result_df["направление"] = result
result_df["направление"] = result_df["направление"].map(label_to_idx)

# In[253]:


import json
data1 = result_df["выход"].tolist()
data2 = result_df["направление"].tolist()
with open('forecast_value.json', 'w') as file:
    json.dump(data1, file)
with open('forecast_class.json', 'w') as file:
    json.dump(data2, file)
