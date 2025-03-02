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

# %%
holiday_df = df[['dteday', 'target', 'holiday', 'holiday_shadow', 'season']]

aggregated_df = holiday_df.groupby('dteday', as_index=False).agg({
    'target': 'sum',
    'holiday': 'first',
    'holiday_shadow': 'first',
    'season': 'first'
}).sort_values(by='dteday').fillna(0)

# %%
import matplotlib.pyplot as plt
import pandas as pd

# Ensure dteday is in datetime format
aggregated_df['dteday'] = pd.to_datetime(aggregated_df['dteday'])

# Filter for the year 2022
df_2022 = aggregated_df[aggregated_df['dteday'].dt.year == 2022]

# Identify holiday dates in 2022
holiday_dates_2022 = df_2022[df_2022['holiday'] == 1]['dteday']
holiday_season_dates_2022 = df_2022[df_2022['holiday_shadow'] == 1]['dteday']  # Get season dates

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df_2022['dteday'], df_2022['target'], linestyle='-', label='Daily Bikes Rented')

# Add green shaded area for holiday season
for season in holiday_season_dates_2022:
    plt.axvspan(season, season + pd.Timedelta(days=1), color='green', alpha=0.3, label='Holiday Season' if 'Holiday Season' not in plt.gca().get_legend_handles_labels()[1] else "")

# Add vertical red lines for holidays
for holiday in holiday_dates_2022:
    plt.axvline(x=holiday, color='red', linestyle='--', alpha=0.7, label='Holiday' if 'Holiday' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.xlabel('Date')
plt.ylabel('Bikes Rented')
plt.title('Bikes Rented Over 2022 with Holidays and Holiday Shadow')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()

# %%
from matplotlib.patches import Patch

# Ensure dteday is in datetime format
aggregated_df['dteday'] = pd.to_datetime(aggregated_df['dteday'])

# Resample the data by month and sum the 'target' for each month
monthly_target = aggregated_df.resample('M', on='dteday')['target'].sum()

# Plot the monthly target sum
plt.figure(figsize=(12, 6))

# Plot the line for monthly target sum
plt.plot(monthly_target.index, monthly_target.values, linestyle='-', color='b', label='Monthly Target Sum')

# Loop through the dataframe to add shaded regions for each season
for i, row in aggregated_df.iterrows():
    if pd.to_datetime(row['dteday']).month != pd.to_datetime(aggregated_df.iloc[i-1]['dteday']).month:
        season = row['season']
        color_map = {1: 'green', 2: 'red', 3: 'blue', 4: 'orange'}
        plt.axvspan(pd.to_datetime(row['dteday']), pd.to_datetime(row['dteday']) + pd.Timedelta(days=30),
                    color=color_map.get(season, 'gray'), alpha=0.3)

# Custom legend for seasons
legend_elements = [
    Patch(color='green', alpha=0.3, label='Winter'),
    Patch(color='red', alpha=0.3, label='Spring'),
    Patch(color='blue', alpha=0.3, label='Summer'),
    Patch(color='orange', alpha=0.3, label='Fall')
]

# Customize plot
plt.xlabel('Month')
plt.ylabel('Bikes Rented')
plt.title('Monthly Rental Sum Over Time with Seasonal Shading')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(handles=legend_elements, loc='upper left')
plt.show()
# %%
