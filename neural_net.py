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
df['covid'] = (df['dteday'].dt.year.isin([2020, 2021])).astype(int)

df['commuting_hours'] = df['dteday'].dt.hour.isin([7, 8, 16, 17]).astype(int)

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

def update_holiday_column(df):
    """Update the 'holiday' column in the dataframe based on US holidays."""
    df['holiday'] = df['dteday'].apply(is_holiday)
    return df

def add_holiday_shadow(df):
    """Create a new column 'holiday_shadow' marking dates within a week of a holiday."""
    
    # Convert 'dteday' to datetime if not already
    df['dteday'] = pd.to_datetime(df['dteday'])
    
    # Get unique dates
    unique_dates = df['dteday'].unique()
    
    # Find all holiday dates
    holiday_dates = set(date for date in unique_dates if is_holiday(date))
    
    # Create a new column initialized with 0
    df['holiday_shadow'] = 0
    
    # Check if each date is within 7 days of a holiday
    for date in unique_dates:
        if any(abs((date - holiday).days) <= 7 for holiday in holiday_dates):
            df.loc[df['dteday'] == date, 'holiday_shadow'] = 1
            
    return df

#%%
df = update_holiday_column(df)
df = add_holiday_shadow(df)

#%%
scaler = MinMaxScaler()
num_features = ['hr', 'temp_c', 'feels_like_c', 'hum', 'windspeed', 'epoch']
df[num_features] = scaler.fit_transform(df[num_features])

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
with tf.device('/GPU:0'):
    # model = Sequential([
    #         Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    #         Dense(16, activation='relu'),
    #         Dropout(0.2),
    #         Dense(32, activation='relu'),
    #         Dropout(0.2),
    #         Dense(1)  # Output layer (no activation for regression)
    #     ])
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),  # First layer
        Dense(256, activation='relu'),  # Second layer with 256 units
        Dropout(0.2),  # Dropout after second layer
        Dense(1)  # Output layer (assuming regression task)
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                loss='mse',  # Mean Squared Error for regression
                metrics=['mae'])  # Mean Absolute Error for better interpretability

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

#%%
# Train the model
with tf.device('/GPU:0'):
    model.fit(X_train, y_train, epochs=100, batch_size=256, 
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




#%%
# # Define the model-building function for Keras Tuner
# def build_model(hp):
#     # Initialize the model
#     model = Sequential()
    
#     # Input layer (first layer) 
#     model.add(Dense(hp.Choice('units_1', [32, 64, 128, 256, 512, 1024]), activation='relu', input_shape=(X_train.shape[1],)))
    
#     # Dynamically add layers based on the number of layers hyperparameter
#     num_layers = hp.Int('num_layers', min_value=2, max_value=6, step=1)  # You can test between 2 to 6 layers now

#     for i in range(2, num_layers + 1):
#         model.add(Dense(hp.Choice(f'units_{i}', [32, 64, 128, 256, 512, 1024]), activation='relu'))
#         model.add(Dropout(hp.Choice(f'dropout_{i}', [0.2, 0.3, 0.4, 0.5, 0.6])))  # More dropout options
    
#     # Output layer (no activation for regression)
#     model.add(Dense(1))  
    
#     # Compile the model
#     model.compile(optimizer=keras.optimizers.Adam(
#                     learning_rate=hp.Choice('learning_rate', [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1])),
#                   loss='mse',
#                   metrics=['mae'])
#     return model

# # Initialize the tuner
# tuner = kt.GridSearch(
#     build_model,
#     objective='val_mae',  # Minimize validation MAE
#     max_trials=100,  # Increased trials to try 10x more combinations
#     executions_per_trial=2,  # Train each configuration multiple times to reduce variance
#     directory='kt_logs',
#     project_name='grid_search'
# )

# # Perform the search
# tuner.search(X_train, y_train, epochs=50, batch_size=128,  # Larger batch size for GPU usage
#              validation_data=(X_test, y_test), verbose=2)

# # Get the best hyperparameters
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# print("Best Hyperparameters:", best_hps.values)

# # Build the model with best hyperparameters
# best_model = tuner.hypermodel.build(best_hps)


# # Train the best model
# best_model.fit(X_train, y_train, epochs=100, batch_size=128,  # Larger batch size for GPU usage
#                validation_data=(X_test, y_test), verbose=2)



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
# holdout = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes_december.csv')
# Ensure 'dteday' column exists in the holdout data and process the 'epoch' column
holdout['dteday'] = pd.to_datetime(holdout['dteday'])
holdout['epoch'] = pd.to_datetime(holdout['dteday'], dayfirst=False).apply(lambda x: int(x.timestamp()))

holdout['covid'] = (holdout['dteday'].dt.year.isin([2020, 2021])).astype(int)

holdout['commuting_hours'] = holdout['dteday'].dt.hour.isin([7, 8, 16, 17]).astype(int)

# Apply the holiday check function
holdout = update_holiday_column(holdout)
holdout = add_holiday_shadow(holdout)

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
save_df.to_csv('./team61-module4-predictions.csv', index=False)
# %%
