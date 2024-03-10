import pandas as pd
import matplotlib.pyplot as plt


def calculate_eman(data, n):
    alpha = 2 / (n+1)
    ema_values = []
    for i in range(len(data)):
        if i == 0:
            ema_values.append(data[i])
        else:
            ema_values.append(alpha*data[i] + (1-alpha) * ema_values[i-1])
    return ema_values


# load data
df = pd.read_csv('data/nvda_us_w.csv')

# creating variables for dartes and closing values
dates = pd.to_datetime(df['Data'])
closing = df['Zamkniecie']

# calculate ema for n=12
macd12 = calculate_eman(closing, 12)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(dates, macd12, label='EMAN (n=12)', color='blue')
plt.title('Exponential Moving Average (EMAn) for n=12')
plt.xlabel('Date')
plt.ylabel('EMAN Values')
plt.legend()
plt.show()
