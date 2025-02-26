#%%
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime

# %%
df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes.csv')

# %%
df['epoch'] = pd.to_datetime(df['dteday'], dayfirst=False).apply(lambda x: int(x.timestamp()))

#%%
scaler = MinMaxScaler()
num_features = ['hr', 'temp_c', 'feels_like_c', 'hum', 'windspeed', 'epoch']
df[num_features] = scaler.fit_transform(df[num_features])

# %%
# Prepare data for 'casual' prediction
df_casual = df.drop(columns=['registered', 'dteday'])
X = df_casual.drop('casual', axis=1)
y = df_casual['casual']

#%%
# df_registered = df.drop(columns=['casual', 'dteday'])
# X = df_casual.drop('casual', axis=1)
# y = df_casual['casual']

#%%
# One-hot encode categorical variables
X = pd.get_dummies(X, columns=['weathersit', 'season', 'holiday', 'workingday'])
# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors='coerce')
X = X.astype(float)
X[X.select_dtypes('bool').columns] = X.select_dtypes('bool').astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to NumPy arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # First hidden layer
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer (no activation for regression)
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss='mse',  # Mean Squared Error for regression
              metrics=['mae'])  # Mean Absolute Error for better interpretability

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

#%%
# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, 
          validation_data=(X_test, y_test), verbose=2, 
          callbacks=[tensorboard_callback])

#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Get predictions on the test data
y_pred = model.predict(X_test)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)
# Print the results
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'R^2: {r2:.4f}')














# %%
holdout = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/biking_holdout_test_mini.csv')

holdout['epoch'] = pd.to_datetime(holdout['dteday'], dayfirst=False).apply(lambda x: int(x.timestamp()))

scaler = MinMaxScaler()
num_features = ['hr', 'temp_c', 'feels_like_c', 'hum', 'windspeed']
holdout[num_features] = scaler.fit_transform(holdout[num_features])