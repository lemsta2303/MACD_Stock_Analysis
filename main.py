import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# load data
df = pd.read_csv('data/nvda_us_w.csv')

# creating variables for dartes and closing values
dates = pd.to_datetime(df['Data'])
closing = df['Zamkniecie']


class Macd:
    __macd = []

    def __init__(self, data):
        self.calculate_macd(data)

    def calculate_ema(self, n, data, day):
        alpha = 2 / (n + 1)
        data_arr = data.array
        p = data_arr[day - n: day + 1:]
        p = np.flip(p)
        counter = 0.0
        denominator = 0.0
        for i in range(n + 1):
            counter += (1 - alpha) ** i * p[i]
            denominator += (1 - alpha) ** i
        return counter / denominator

    def calculate_macd(self, data):
        self.__macd = []
        for i in range(len(data)):
            if i >= 26:
                ema12 = self.calculate_ema(12, data, i)
                ema26 = self.calculate_ema(26, data, i)
                temp_macd = ema12 - ema26
                self.__macd.append(temp_macd)
            else:
                self.__macd.append(float(0))

    def get_macd(self):
        return self.__macd


nvidia_macd = Macd(closing)

plt.figure(figsize=(12, 6))
plt.plot(dates[26:], closing[26:], label='NVIDIA STOCK', color="red")
plt.plot(dates[26:], nvidia_macd.get_macd()[26:], label='NVIDIA MACD', color='blue')
plt.title('NVIDIA Stock Closing Values and MACD')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()