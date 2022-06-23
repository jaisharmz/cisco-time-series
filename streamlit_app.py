import os
os.system("pip install kats")

import streamlit as st
import pkg_resources
import numpy as np
import pandas as pd
import sklearn
import time
import math
import statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
# from prophet import Prophet
# from prophet.plot import add_changepoints_to_plot
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, LSTM
import kats
from kats.consts import TimeSeriesData
from kats.models import *
from kats.models.ensemble.kats_ensemble import *
from kats.models.metalearner.get_metadata import GetMetaData
from kats.models.metalearner.metalearner_predictability import MetaLearnPredictability
from kats.models.metalearner.metalearner_modelselect import MetaLearnModelSelect
import pickle
from IPython.display import display

tf.random.set_seed(0)
np.random.seed(0)

# # Generator for synthetic data
# generator = synthesis([
#     value(100),
#     slope(0.2),
#     noise(0, 5),
#     noise(0, 10),
#     noise(0, 20),
#     noise(0, 20),
#     spike(100,10,100,10,10,2),
#     spike(100,10,100,10,10,2),
#     spike(100,10,100,10,10,2),
#     spike(100,10,100,10,10,2),
#     level_shift(100,10,50,10),
#     level_shift(100,10,-50,10)
# ])
#
# np.random.seed(1) # Seed
# num_examples = 10000 # Number of examples
# df = pipeline(generator, num_examples, return_type="pandas", out_file="nums.csv") # Synthesize data
# # Make time column
# time_col = [int(time.time())]
# for i in range(len(df["time_step"])-1):
#   time_col.append(time_col[-1] + max(1, 1000 + np.random.normal(100, 10)))
# df["time"] = [pd.Timestamp(i, unit='s') for i in time_col]
# df["time_ms"] = np.array(df["time"].astype(np.int64) / 10**6, np.int64)

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def find_arange(x):
  x = np.array(x)
  return np.arange(x.size).reshape(x.shape)

def pairwise_diff(x):
  x = np.array(x)
  return x[1:] - x[:-1]

def find_outliers(values, threshold=2):
  z_scores = (values - np.mean(values)) / np.std(values)
  num_outliers = np.sum((z_scores > threshold) + (z_scores < -threshold))
  indices_no_outliers = ((z_scores < threshold) & (z_scores > -threshold)).astype(bool)
  lower = -threshold * np.std(values) + np.mean(values)
  upper = threshold * np.std(values) + np.mean(values)

  return values, indices_no_outliers, num_outliers, lower, upper

def synthesis(layers):
  '''
  Synthesizes the random layers as given by the parameter layers.
  @param: layers is a list of random layers, each of which are generator functions.
  '''
  time_step = 0
  while True:
    # update value
    value = 0
    # update value with no anomaly (na)
    value_na = 0
    # is there an anomaly? (generating labels)
    anomalous = False
    for i in layers:
      next_num = next(i)
      value += next_num
      if not(i.__name__ == "spike") and not(i.__name__ == "level_shift"):
        value_na += next_num
      elif next_num != 0 and i.__name__ == "spike":
        anomalous = True

    # update time step
    time_step += 1

    yield [time_step, float(value_na), float(value), anomalous]

def value(value):
  '''
  @param: value is y intercept.
  '''
  while True:
    yield value

def slope(slope):
  '''
  @param: slope adds a slight slope to the overall data.
  '''
  value = 0
  while True:
    yield value
    value += slope

def noise(mu, sigma):
  '''
  Generates statistical noise using mean and standard deviation.
  @param: mu is the mean of the noise.
  @param: sigma is the standard deviation of the noise.
  '''
  while True:
    yield np.random.normal(mu,sigma,(1,1))

def spike(period_mu, period_sigma, spike_mu, spike_sigma, length_mu, length_sigma):
  '''
  Provides spike anomalies to the data.
  @param: period_mu is the mean of the period between spikes.
  @param: period_sigma is the standard deviation of the period between spikes.
  @param: spike_mu is the mean of the spike magnitude.
  @param: spike_sigma is the standard deviation of the spike magnitude.
  @param: length_mu is the mean of the duration of the spike anomaly.
  @param: length_sigma is the standard deviation of the duration of the spike anomaly.
  '''
  spike = False
  value = 0
  while True:
    if spike and np.random.random() < 1 - 0.5 ** (1 / np.random.normal(length_mu,length_sigma,(1,1))):
      value = 0
      spike = False
    if not(spike) and np.random.random() < 1 - 0.5 ** (1 / np.random.normal(period_mu,period_sigma,(1,1))):
      value += np.random.normal(spike_mu,spike_sigma,(1,1))
      spike = True
    yield value

def level_shift(period_mu, period_sigma, shift_mu, shift_sigma):
  '''
  Provides level shift anomalies to the data.
  @param: period_mu is the mean of the period between level shifts.
  @param: period_sigma is the standard deviation of the period between level shifts.
  @param: shift_mu is the mean of the shift magnitude.
  @param: shift_sigma is the standard deviation of the shift magnitude.
  '''
  value = 0
  while True:
    if np.random.random() < 1 - 0.5 ** (1 / np.random.normal(period_mu,period_sigma,(1,1))):
      value += np.random.normal(shift_mu,shift_sigma,(1,1))
    yield value

def pipeline(generator, num_time_steps, return_type="numpy", out_file=None):
  '''
  Pipeline is used to create a data set of synthesized random numbers based on a generator function.
  @param: generator is the generator function which will be used to generate the synthetic data.
  @param: num_time_steps is the number of time steps the user desires to generate data for.
  @param: return_type is the return type of the synthetic data. Choices are "numpy," "list," or "pandas."
  @param: out_file is the name of the file to which this function should output the generated data.
  '''
  nums = []
  for i in range(num_time_steps):
    nums.append(next(generator))

  if return_type == "numpy":
    nums = np.array(nums)
    if out_file:
      pickle.dump(nums, open(out_file, "wb"))
  elif return_type == "pandas":
    nums = pd.DataFrame(np.array(nums), columns=["time_step", "value_na", "value", "anomalous"])
    if out_file:
      nums.to_csv(out_file)
  elif out_file:
    pickle.dump(nums, open(out_file, "wb"))

  return nums

def quality_score(data, time_col=None):
  try:
    if time_col:
      total_time = data[time_col].iloc[-1] - data[time_col].iloc[0]
      median_pairwise_time = np.median(pairwise_diff(data[time_col]))
      expected_points = total_time / median_pairwise_time + 1
      true_points = data.shape[0]
      quality_score = 100 * true_points / expected_points
    else:
      total_time = data[-1,0] - data[0,0]
      median_pairwise_time = np.median(pairwise_diff(data[:,0]))
      expected_points = total_time / median_pairwise_time + 1
      true_points = data.shape[0]
      quality_score = 100 * true_points / expected_points
    return quality_score
  except:
    raise Exception("Score is undetermined because column is invalid.")

def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, y_data.shape[0], batch_size)
    return x_data[idxs,:,:], y_data[idxs]

class TSModel():
  def __init__(self, name="lstm", num_units=4, lookback=10, pred_length=1):
    self.name = name
    self.num_units = num_units
    self.lookback = lookback
    self.pred_length = pred_length
    self.model = self.make_model(name)
  def make_model(self, name):
    if name == "lstm":
      model = Sequential([
                    LSTM(self.num_units,input_shape=(1,self.lookback)),
                    Dense(self.pred_length),
              ])
    return model
  def fit(self, X_train, y_train, X_test=-1, y_test=-1, epochs=1000, batch_size=None, loss="mae", metrics=["mse"], optimizer="adam", print_metrics=True):
    if self.name == "lstm":
      self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
      if type(X_test)!=int and type(y_test)!=int:
        history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
      else:
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
      if print_metrics:
          history_pd = pd.DataFrame(history.history)
          history_pd = pd.DataFrame(history.history)
          fig, ax = plt.subplots()
          ax.plot(history_pd["loss"], label="Train Set Loss")
          if type(X_test)!=int and type(y_test)!=int:
            ax.plot(history_pd["val_loss"], label="Test Set Loss")
          ax.legend()
          ax.set_xlabel("Epoch")
          ax.set_ylabel(loss)
          if type(X_test)!=int and type(y_test)!=int:
            ax.set_title("Train and Test Set Loss")
          else:
            ax.set_title("Train Set Loss")
          st.pyplot(fig)
          temp = pd.DataFrame(history_pd.iloc[-1,:])
          temp.columns = ["Final Loss and Metrics"]
          st.write(temp)
      return history
  def predict(self, X_vals):
    preds = self.model.predict(X_vals)
    return preds

uploaded_file = st.file_uploader("Upload Data")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file)
    st.write("First 1000 rows displayed:")
    st.write(df.head(1000))

time_col = st.text_input("Time Column Name (Leave Blank for Index)", "time")

data_col = st.text_input("Data Column Name", "value")

model_type = st.selectbox(
                "Model to Use",
                ("LSTM", "Facebook Prophet", "Theta Model", "Holt-Winters", "SARIMA", "ensemble"))

# if model_type == "LSTM":
#     pred_length = st.text_input("Prediction Length (Time stamps to predict)", 1000)
#
#     smoothen_factor = st.text_input("Smoothen Factor (Number in the interval (0,1]; proportion of original data left) (e.g. 0.3 means 30% of original data is left)", 1)
#
#     pred_length_lstm = st.text_input("Prediction Length for LSTM", 100)
#
#     lookback = st.text_input("Lookback for LSTM", 10)
#
#     degree = st.text_input("Overall Polynomial Degree of Data", 1)
#
#     epochs = st.text_input("Epochs (Number of epochs to train resid)", 1000)
if model_type == "LSTM":
    pred_length = st.text_input("Prediction Length (Time stamps to predict)", 1000)

    smoothen_factor = st.text_input("Smoothen Factor (Number in the interval (0,1]; proportion of original data left) (e.g. 0.3 means 30% of original data is left)", 1)

    lookback = st.text_input("Lookback for LSTM", 10)

    units = st.text_input("Number of hidden units", 8)

    epochs = st.text_input("Epochs (Number of epochs to train resid)", 1000)
elif model_type == "Facebook Prophet":
    pred_length = st.text_input("Prediction Length (Time stamps to predict)", 1000)

    smoothen_factor = st.text_input("Smoothen Factor (Number in the interval (0,1]; proportion of original data left) (e.g. 0.3 means 30% of original data is left)", 1)

    seasonality_mode = st.selectbox("Seasonality Mode",
                                    ("additive", "multiplicative"))
elif model_type == "Theta Model":
    pred_length = st.text_input("Prediction Length (Time stamps to predict)", 1000)

    smoothen_factor = st.text_input("Smoothen Factor (Number in the interval (0,1]; proportion of original data left) (e.g. 0.3 means 30% of original data is left)", 1)
elif model_type == "Holt-Winters":
    pred_length = st.text_input("Prediction Length (Time stamps to predict)", 1000)

    smoothen_factor = st.text_input("Smoothen Factor (Number in the interval (0,1]; proportion of original data left) (e.g. 0.3 means 30% of original data is left)", 1)
elif model_type == "SARIMA":
    pred_length = st.text_input("Prediction Length (Time stamps to predict)", 1000)

    smoothen_factor = st.text_input("Smoothen Factor (Number in the interval (0,1]; proportion of original data left) (e.g. 0.3 means 30% of original data is left)", 1)
elif model_type == "ensemble":
    pred_length = st.text_input("Prediction Length (Time stamps to predict)", 1000)

    smoothen_factor = st.text_input("Smoothen Factor (Number in the interval (0,1]; proportion of original data left) (e.g. 0.3 means 30% of original data is left)", 1)
if st.button("Start Analysis"):
    # if model_type == "LSTM":
    #     pred_length = int(pred_length)
    #     smoothen_factor = float(smoothen_factor)
    #     epochs = int(epochs)
    #     degree = int(degree)
    #     pred_length_lstm = int(pred_length_lstm)
    #     lookback = int(lookback)
    if model_type == "LSTM":
        pred_length = int(pred_length)
        smoothen_factor = float(smoothen_factor)
        epochs = int(epochs)
        units = int(units)
        lookback = int(lookback)
    elif model_type == "Facebook Prophet":
        pred_length = int(pred_length)
        smoothen_factor = float(smoothen_factor)
    elif model_type == "Theta Model":
        pred_length = int(pred_length)
        smoothen_factor = float(smoothen_factor)
    elif model_type == "Holt-Winters":
        pred_length = int(pred_length)
        smoothen_factor = float(smoothen_factor)
    elif model_type == "SARIMA":
        pred_length = int(pred_length)
        smoothen_factor = float(smoothen_factor)
    elif model_type == "ensemble":
        pred_length = int(pred_length)
        smoothen_factor = float(smoothen_factor)

    df[time_col] = pd.to_datetime(df[time_col])
    df["time_ms"] = np.array(df[time_col].astype(np.int64) / 10**6, np.int64)

    # Plot anomalous and non-anomalous data
    st.line_chart(df.loc[:,[data_col]])

    # Quality Score
    st.write("Quality Score: ")
    st.write(round(quality_score(df, time_col), 5))


    # Pairwise differencing histogram
    fig, ax = plt.subplots()
    ax.hist(pairwise_diff(df["time_ms"])/1000, bins=200)
    ax.set_title("Pairwise Differences in Timestamps Histogram")
    ax.set_xlabel("Pairwise Difference in Timestamps")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Boxplot
    fig, ax = plt.subplots()
    ax.boxplot(pairwise_diff(df["time_ms"])/1000)
    ax.set_title("Pairwise Differences in Timestamps Boxplot")
    ax.set_xlabel("Time Series Index")
    ax.set_ylabel("Pairwise Difference in Timestamps")
    st.pyplot(fig)

    # Seasonal Decomposition
    minute_ms = 60 * 1000
    freq_minute = int(minute_ms / np.median(pairwise_diff(df["time_ms"])))
    hour_ms = 60 * 60 * 1000
    freq_hour = int(hour_ms / np.median(pairwise_diff(df["time_ms"])))
    day_ms = 24 * 60 * 60 * 1000
    freq_day = int(day_ms / np.median(pairwise_diff(df["time_ms"])))
    week_ms = 7 * day_ms
    freq_week = int(week_ms / np.median(pairwise_diff(df["time_ms"])))
    month_ms = 30 * day_ms
    freq_month = int(month_ms / np.median(pairwise_diff(df["time_ms"])))
    year_ms = 365 * day_ms
    freq_year = int(year_ms / np.median(pairwise_diff(df["time_ms"])))

    freqs = np.array([freq_minute, freq_hour, freq_day, freq_week, freq_month, freq_year])
    freqs_types = np.array(["minute", "hour", "day", "week", "month", "year"])
    periods = df.shape[0] / (freqs + 1e-8)

    freqs_filt = []
    freqs_types_filt = []
    total_time = df["time_ms"].iloc[-1] - df["time_ms"][0]
    # if total_time <= minute_ms * 120:
    #   freqs_filt.append(freq_minute)
    #   freqs_types_filt.append("minute")
    if total_time <= hour_ms * 48 and total_time >= hour_ms * 2:
      freqs_filt.append(freq_hour)
      freqs_types_filt.append("hour")
    if total_time <= day_ms * 14 and total_time >= day_ms * 2:
      freqs_filt.append(freq_day)
      freqs_types_filt.append("day")
      freqs_filt.append(freq_hour)
      freqs_types_filt.append("hour")
    if total_time <= week_ms * 8 and total_time >= week_ms * 2:
      freqs_filt.append(freq_week)
      freqs_types_filt.append("week")
      freqs_filt.append(freq_day)
      freqs_types_filt.append("day")
    if total_time <= month_ms * 24 and total_time >= month_ms * 2:
      freqs_filt.append(freq_month)
      freqs_types_filt.append("month")
      freqs_filt.append(freq_week)
      freqs_types_filt.append("week")
    if total_time >= year_ms * 2:
      freqs_filt.append(freq_year)
      freqs_types_filt.append("year")
      freqs_filt.append(freq_month)
      freqs_types_filt.append("month")

    freqs_filt = np.array(freqs_filt)
    freqs_types_filt = freqs_filt[freqs_filt != 0]
    freqs_filt = freqs_filt[freqs_filt != 0]
    freqs_types_filt = np.array(freqs_types_filt)
    periods_filt = df.shape[0] / (freqs_filt + 1e-8)
    data_rolling_avg = df[data_col].rolling(1+int(df.shape[0]*(1-smoothen_factor))).mean()

    st.write("Moving Average with window size " + str(1+int(df.shape[0]*(1-smoothen_factor))))
    data_avg_plot = pd.DataFrame({"Original Data": df[data_col].values, "Moving Average": data_rolling_avg})
    st.line_chart(data_avg_plot)

    data_rolling_avg = data_rolling_avg.dropna()

    sds = []

    for i in range(len(freqs_filt)):
      temp = data_rolling_avg
      for ii in range(len(sds)):
        temp -= sds[ii].seasonal
      # sds.append(statsmodels.tsa.seasonal.seasonal_decompose(temp,freq=freqs_filt[i]))
      sds.append(statsmodels.tsa.seasonal.seasonal_decompose(temp,period=int(df.shape[0]/periods_filt[i])))

    fig, axs = plt.subplots(nrows=len(freqs_filt)+2,sharex=True,constrained_layout=True)
    axs[0].plot(sds[-1].trend)
    axs[0].set_title("Trend")

    for i in range(len(freqs_filt)-1,-1,-1):
      axs[len(freqs_filt)-i].plot(sds[i].seasonal)
      axs[len(freqs_filt)-i].set_title("Seasonality with Frequency " + str(freqs_filt[i]))

    axs[-1].plot(sds[-1].resid)
    axs[-1].set_title("Residuals")

    # fig.savefig("seasonality.png")
    st.pyplot(fig)

    # Forecasting
    # if model_type == "LSTM":
    #     # Basic parameters
    #     # lookback = 100
    #     # pred_length_lstm = 10
    #     # pred_length = 1000
    #
    #     # Trend
    #     # Make X, y, train, and test sets
    #     st.write("Training predictive model for Trend. Please wait a few moments...")
    #
    #     y = np.array(sds[-1].trend).reshape(-1,1)
    #     y[np.isnan(y)] = 0
    #     X = np.arange(y.shape[0]).reshape(-1,1)
    #
    #     train_size = 0.9
    #     X_train = X[:int(X.shape[0]*train_size)]
    #     X_test = X[int(X.shape[0]*train_size):]
    #     y_train = y[:int(y.shape[0]*train_size)]
    #     y_test = y[int(y.shape[0]*train_size):]
    #
    #     # Build and fit polynomial regression model
    #     polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())
    #     polyreg.fit(X_train,y_train)
    #
    #     # Make predictions
    #     st.write("Done training predictive model for Trend. Making predictions using this model for Trend. Please wait a few moments...")
    #     trend_pred = polyreg.predict(np.arange(y.shape[0] + pred_length).reshape(-1,1))
    #
    #     # Plot predictions and original trend together
    #     # index = list(sds[-1].trend.index)
    #     # index += [index[-1]+i+1 for i in range(pred_length)]
    #     # trend_pred_plot = pd.DataFrame({"Original Trend": sds[-1].trend, "Predicted Trend": pd.Series(trend_pred.reshape(-1), index=index)})
    #     # st.write("Predicted Trend")
    #     # st.line_chart(trend_pred_plot)
    #
    #     # Resid
    #     y = np.array(sds[-1].resid).reshape(-1,1)
    #     y[np.isnan(y)] = 0
    #
    #     # Make X, y, train, and test sets
    #     X_lstm = []
    #     y_lstm = []
    #
    #     for i in range(y.shape[0]-lookback):
    #       X_lstm.append(y[i:i+lookback].reshape(-1))
    #       y_lstm.append(y[i+lookback].reshape(-1))
    #
    #     X_lstm = np.array(X_lstm)
    #     X_lstm = X_lstm.reshape(X_lstm.shape[0],1,X_lstm.shape[1])
    #     y_lstm = np.array(y_lstm)
    #
    #     train_size = 0.7
    #     X_lstm_train = X_lstm[:int(X_lstm.shape[0]*train_size)]
    #     X_lstm_test = X_lstm[int(X_lstm.shape[0]*train_size):]
    #     y_lstm_train = y_lstm[:int(y_lstm.shape[0]*train_size)]
    #     y_lstm_test = y_lstm[int(y_lstm.shape[0]*train_size):]
    #
    #     # Set random seeds
    #     tf.random.set_seed(0)
    #     np.random.seed(0)
    #
    #     st.write("Training predictive model for Resid. Please wait a few moments...")
    #
    #     # Build and train model
    #     model = TSModel(name="lstm", num_units=lookback, lookback=lookback, pred_length=pred_length_lstm)
    #     history = model.fit(X_lstm_train, y_lstm_train, X_lstm_test, y_lstm_test, epochs=epochs, batch_size=128, loss="mae", metrics=["mse"])
    #
    #     st.write("Done training predictive model for Resid. Making predictions using this model for Resid. Please wait a few moments...")
    #
    #     last_index = y.shape[0] - 1
    #     while (y==0)[last_index]:
    #       last_index -= 1
    #     resid_pred = y.tolist()[:last_index+1]
    #
    #     for i in range(math.ceil((y.shape[0]-last_index+pred_length)/pred_length_lstm)):
    #       pred = model.predict(np.array(resid_pred)[last_index+1-lookback:last_index+1].reshape(-1,1,lookback))
    #       resid_pred += [[i] for i in pred.tolist()[0]]
    #       last_index = len(resid_pred) - 1
    #     resid_pred = np.array(resid_pred)[:y.shape[0]+pred_length]
    #
    #     # Plot
    #     # st.write("Predicted Resid")
    #     # plt.plot(resid_pred)
    #     # plt.plot(y)
    #     # plt.show()
    #
    #     # Seasonal
    #     st.write("Making predictions for seasonality of data. Please wait a few moments...")
    #     seasonal_pred = np.array([sum([sds[i].seasonal.iloc[j%freqs_filt[i]] for i in range(len(sds))]) for j in range(y.shape[0]+pred_length)]).reshape(-1,1)
    #
    #     # Plot
    #     # plt.plot(seasonal_pred)
    #
    #     # Putting them together!
    #     pred = pd.DataFrame(resid_pred + trend_pred + seasonal_pred)
    #     index = list(data_rolling_avg.index)
    #     index = index + [index[-1] + i + 1 for i in range(pred_length)]
    #     pred.index = index
    #
    #     st.write("Final Predictions")
    #     pred_plot = pd.DataFrame({"Original Moving Average Data": pd.Series(data_rolling_avg, index=pred.index), "Prediction": pred.values.reshape(-1)})
    #     pred_plot.index = pred.index
    #     st.line_chart(pred_plot)
    #
    #     # Output to CSV
    #     pred_pd = pd.DataFrame(pred)
    #     pred_pd.columns = ["Prediction"]
    #     pred_compact = convert_df(pred_pd)
    #
    #     st.download_button(
    #      label="Download Predictions as CSV",
    #      data=pred_compact,
    #      file_name='predictions.csv',
    #      mime='text/csv',
    #      )
    # data_rolling_avg = df[data_col].rolling(int(df.shape[0]/smoothen_factor)).mean().dropna()
    ts = pd.DataFrame({"ds": df.iloc[data_rolling_avg.index, :][time_col], "y": data_rolling_avg})
    ts.index = ts["ds"]

    ts_kats = TimeSeriesData(time=ts["ds"], value=ts["y"])

    if model_type == "LSTM":
        params = lstm.LSTMParams(hidden_size=units,
                    time_window=lookback,
                    num_epochs=epochs)
        model = lstm.LSTMModel(ts_kats, params)
        model.fit()
        pred = model.predict(steps=pred_length)
        pred.index = ts_kats.to_dataframe().index[-1] + pred.index + 1

    elif model_type == "Facebook Prophet":
        params = prophet.ProphetParams(seasonality_mode=seasonality_mode)
        model = prophet.ProphetModel(ts_kats, params)
        model.fit()
        pred = model.predict(steps=pred_length, freq="MS")
        pred.index = ts_kats.to_dataframe().index[-1] + pred.index + 1

        # # instantiate the model and fit the timeseries
        # prophet = Prophet()
        # prophet.fit(ts)
        #
        # # create a future data frame
        # future = prophet.make_future_dataframe(periods=periods_pred)
        # forecast = prophet.predict(future)
        #
        # # plot
        # fig = prophet.plot(forecast)
        # st.write("Facebook Prophet Fitted to Data")
        # st.pyplot(fig)
        #
        # a = add_changepoints_to_plot(fig.gca(),prophet,forecast)
        # st.write("Changepoints Detected by Facebook Prophet")
        # st.pyplot(fig)
        #
        # # Output to CSV
        # pred_compact = forecast[['ds','yhat','yhat_lower','yhat_upper']]
        # pred_compact = convert_df(pred_compact)

    elif model_type == "Theta Model":
        params = theta.ThetaParams(m=freqs_filt[0])
        model = theta.ThetaModel(ts_kats, params)
        model.fit()
        pred = model.predict(steps=pred_length)

    elif model_type == "Holt-Winters":
        params = holtwinters.HoltWintersParams(
                                    trend="add",
                                    #damped=False,
                                    seasonal="mul",
                                    seasonal_periods=12,
                                )
        model = holtwinters.HoltWintersModel(ts_kats, params)
        model.fit()
        pred = model.predict(steps=pred_length, alpha = 0.95)

    elif model_type == "SARIMA":
        params = sarima.SARIMAParams(
                            p = 2,
                            d=1,
                            q=1,
                            trend = 'ct',
                            seasonal_order=(1,0,1,12)
                            )
        model = sarima.SARIMAModel(ts_kats, params)
        model.fit()
        pred = model.predict(steps=pred_length)

    elif model_type == "ensemble":
        params = kats.models.ensemble.ensemble.EnsembleParams(
                                [
                                    ensemble.ensemble.BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
                                    ensemble.ensemble.BaseModelParams(
                                        "sarima",
                                        sarima.SARIMAParams(
                                            p=2,
                                            d=1,
                                            q=1,
                                            trend="ct",
                                            seasonal_order=(1, 0, 1, 12),
                                            enforce_invertibility=False,
                                            enforce_stationarity=False,
                                        ),
                                    ),
                                    ensemble.ensemble.BaseModelParams("prophet", prophet.ProphetParams()),  # requires fbprophet be installed
                                    ensemble.ensemble.BaseModelParams("linear", linear_model.LinearModelParams()),
                                    ensemble.ensemble.BaseModelParams("quadratic", quadratic_model.QuadraticModelParams()),
                                    ensemble.ensemble.BaseModelParams("theta", theta.ThetaParams(m=12)),
                                ]
                            )
        KatsEnsembleParam = {
                                    "models": params,
                                    "aggregation": "median",
                                    "seasonality_length": 12,
                                    "decomposition_method": "multiplicative",
                                }
        model = KatsEnsemble(data=ts_kats, params=params)
        model.fit()
        pred = model.predict(steps=pred_length)

    fig, ax = plt.subplots()
    ax.plot(ts_kats.to_dataframe().index, ts_kats.to_dataframe()["y"])
    ax.plot(pred.index, pred["fcst"], color='blue')
    ax.fill_between(pred.index, pred["fcst_lower"], pred["fcst_upper"], color='blue', alpha=0.5)
    st.pyplot(fig)

    pred_compact = convert_df(pred)
    st.download_button(
     label="Download Predictions as CSV",
     data=pred_compact,
     file_name='predictions.csv',
     mime='text/csv',
     )

    # # Make another time series with constant frequency
    # df_temp = ts_kats.to_dataframe()
    # df_temp.columns = ["time", "value"]
    # temp = time.time()
    # for i in range(df_temp.shape[0]):
    #     df_temp.iloc[i,0] = temp + 1000 * i
    # df_temp["time"] = pd.to_datetime(df_temp["time"])
    # meta_kats = TimeSeriesData(df_temp)
    #
    # # Get Metadata for models to be trained
    # MD = GetMetaData(data=meta_kats, error_method='mape', num_trials=1)
    # # print(MD.all_models)
    # # print(MD.all_params)
    # # hpt_res = MD.tune_executor()
    # # print(hpt_res)
    # metadata = MD.get_meta_data()
    # metadata_df = pd.DataFrame(metadata)
    # metadata_list = metadata#_df.to_dict(orient='records')
    # st.write(str(metadata))
    # print(metadata_df)
    #
    # # # Predictability of data
    # # mlp = MetaLearnPredictability(metadata_list, threshold=0.2)
    # # st.write(mlp.train())
    # # st.write(mlp.pred(air_passengers_ts))
    # # mlp.save_model("mlp.pkl") # Save predictibility model
    #
    # # # Model Selection
    # # mlms = MetaLearnModelSelect(metadata_list)
    # # st.write(mlms.count_category())
    # # mlms.preprocess(downsample=False, scale=True)
    # # results=mlms.train()
    # # st.write(results=mlms.train())
    # # results_df=pd.DataFrame([results['fit_error'], results['pred_error']])
    # # results_df['error_type']=['fit_error', 'pred_error']
    # # results_df['error_metric']='MAPE'
    # # st.write(results_df)
    # # st.write(mlms.pred(meta_kats))
    # # mlms.save_model("mlms.pkl")
    #
    # #
    #
    #
    #
    #
    #
    #
    # # # Anomaly Detection
    # # anom_model = kats.detectors.cusum_detection.CUSUMDetector(ts_kats)
    # # change_points = anom_model.detector(return_all_changepoints=True)
    # # fig, ax = plt.subplots()
    # # for i in range(len(change_points)):
    # #     ax.axvline(x = change_points[i][0].start_time, color = "red")
    # # ax.plot(ts_kats.to_dataframe()["ds"], ts_kats.to_dataframe()["y"])
    # # st.pyplot(fig)
