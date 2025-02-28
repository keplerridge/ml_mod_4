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
import keras
import keras_tuner as kt

# %%
df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes.csv')
df['dteday'] = pd.to_datetime(df['dteday'])

#%%
df['target'] = df['casual'] + df['registered']

# %%
df['epoch'] = pd.to_datetime(df['dteday'], dayfirst=False).apply(lambda x: int(x.timestamp()))

#%%
def is_holiday(date):
    """Check if a given date is a US holiday from the provided list."""
    year = date.year
    month = date.month
    day = date.day
    weekday = date.weekday()  # Monday = 0, Sunday = 6
    
    # Static holidays
    fixed_holidays = {
        (1, 1),   # January 1st
        (1, 20),  # January 20th
        (2, 17),  # February 17th
        (6, 19),  # June 19th
        (7, 4),   # July 4th
        (12, 24), # December 24th
        (12, 25), # December 25th
        (12, 31)  # December 31st
    }
    
    # Check fixed date holidays
    if (month, day) in fixed_holidays:
        return 1
    
    # Memorial Day (last Monday in May)
    if month == 5 and weekday == 0:  # Monday
        last_monday = max(d for d in range(25, 32) if datetime.date(year, 5, d).weekday() == 0)
        if day == last_monday:
            return 1
    
    # Columbus Day (second Monday in October)
    if month == 10 and weekday == 0:  # Monday
        second_monday = sorted(d for d in range(1, 15) if datetime.date(year, 10, d).weekday() == 0)[1]
        if day == second_monday:
            return 1
    
    # Thanksgiving (fourth Thursday in November)
    if month == 11 and weekday == 3:  # Thursday
        fourth_thursday = sorted(d for d in range(22, 30) if datetime.date(year, 11, d).weekday() == 3)[0]
        if day == fourth_thursday:
            return 1
        
    # Black Friday (day after Thanksgiving)
    if month == 11 and weekday == 4:  # Friday
        thanksgiving = sorted(d for d in range(22, 30) if datetime.date(year, 11, d).weekday() == 3)[0]
        if day == thanksgiving + 1:
            return 1
    
    # December 26 (if it's a Friday)
    if month == 12 and day == 26 and weekday == 4:  # Friday
        return 1
    
    return 0

def update_holiday_column(df):
    """Update the 'holiday' column in the dataframe based on US holidays."""
    df['holiday'] = df['dteday'].apply(is_holiday)
    return df

#%%
df = update_holiday_column(df)

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
df_target = df.drop(columns=['registered', 'casual', 'dteday'])
X = df_target.drop(columns=['target'])
y = df_target['target']

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

#%%
# Define the model
model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)  # Output layer (no activation for regression)
    ])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
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




# #%%
# # Define the model-building function for Keras Tuner
# def build_model(hp):
#     model = Sequential([
#         Dense(hp.Choice('units_1', [32, 64, 128]), activation='relu', input_shape=(X_train.shape[1],)),
#         Dense(hp.Choice('units_2', [16, 32, 64]), activation='relu'),
#         Dropout(hp.Choice('dropout_1', [0.2, 0.5])),
#         Dense(hp.Choice('units_3', [32, 64, 128]), activation='relu'),
#         Dropout(hp.Choice('dropout_2', [0.2, 0.5])),
#         Dense(1)  # Output layer (no activation for regression)
#     ])
    
#     model.compile(optimizer=keras.optimizers.Adam(
#                     learning_rate=hp.Choice('learning_rate', [0.001, 0.0001, 0.01])),
#                   loss='mse',
#                   metrics=['mae'])
#     return model

# # Initialize the tuner
# tuner = kt.GridSearch(
#     build_model,
#     objective='val_mae',  # Minimize validation MAE
#     max_trials=10,  # Number of different hyperparameter combinations to try
#     executions_per_trial=2,  # Train each configuration multiple times to reduce variance
#     directory='kt_logs',
#     project_name='grid_search'
# )

# # Perform the search
# tuner.search(X_train, y_train, epochs=50, batch_size=32,
#              validation_data=(X_test, y_test), verbose=2)

# # Get the best hyperparameters
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# print("Best Hyperparameters:", best_hps.values)

# # Build the model with best hyperparameters
# best_model = tuner.hypermodel.build(best_hps)

# # Train the best model
# best_model.fit(X_train, y_train, epochs=100, batch_size=32, 
#                validation_data=(X_test, y_test), verbose=2)


# #%%
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np

# # Get predictions on the test data
# y_pred = best_model.predict(X_test)

# # Mean Absolute Error (MAE)
# mae = mean_absolute_error(y_test, y_pred)

# # Mean Squared Error (MSE)
# mse = mean_squared_error(y_test, y_pred)

# # Root Mean Squared Error (RMSE)
# rmse = np.sqrt(mse)

# r2 = r2_score(y_test, y_pred)
# # Print the results
# print(f'Mean Absolute Error (MAE): {mae:.4f}')
# print(f'Mean Squared Error (MSE): {mse:.4f}')
# print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
# print(f'R^2: {r2:.4f}')




























# %%
holdout = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/biking_holdout_test_mini.csv')

# Ensure 'dteday' column exists in the holdout data and process the 'epoch' column
holdout['dteday'] = pd.to_datetime(holdout['dteday'])
holdout['epoch'] = pd.to_datetime(holdout['dteday'], dayfirst=False).apply(lambda x: int(x.timestamp()))

# Apply the holiday check function
holdout = update_holiday_column(holdout)

# Apply the same scaling to the numeric columns
holdout[num_features] = scaler.transform(holdout[num_features])

# One-hot encode categorical variables (use the same columns as the training set)
holdout = pd.get_dummies(holdout, columns=['weathersit', 'season', 'holiday', 'workingday'])

# Ensure all features are numeric
holdout = holdout.apply(pd.to_numeric, errors='coerce')
holdout = holdout.astype(float)
holdout[holdout.select_dtypes('bool').columns] = holdout.select_dtypes('bool').astype(int)

# Make sure that the same columns are present in the holdout data as in the training data
# Align columns by adding any missing columns with zeros
missing_cols = set(X.columns) - set(holdout.columns)
for col in missing_cols:
    holdout[col] = 0

# Reorder columns to match the training set
holdout = holdout[X.columns]

# Convert to NumPy array
holdout = holdout.to_numpy()

#%%
# Run the holdout data through the model
holdout_predictions = model.predict(holdout)

# %%
save_df = pd.DataFrame(holdout_predictions, columns=['predictions'])

# %%
save_df.to_csv('./team6-module4-predictions.csv', index=False)
# %%
