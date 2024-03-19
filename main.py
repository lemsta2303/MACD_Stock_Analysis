import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# load data
df = pd.read_csv('data/btc_v_d.csv')[-1000:]

# creating variables for dartes and closing values
dates = pd.to_datetime(df['Data'])
closing = df['Zamkniecie']


class Macd:
    __macd = []
    __signal = []

    def __init__(self, data):
        self.calculate_macd(data)
        self.calculate_signal()

    def calculate_ema(self, n, data, day):
        alpha = 2 / (n + 1)

        # sprawdzenie czy data jest tablica, jesli nie to konwersja na tablcie
        if isinstance(data, np.ndarray):
            data_arr = data
        else:
            data_arr = np.array(data)

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

    def calculate_signal(self):
        macd_data = self.get_macd()
        self.__signal = []
        for i in range(len(macd_data)):
            if i >= 9:
                ema9 = self.calculate_ema(9, macd_data, i)
                self.__signal.append(ema9)
            else:
                self.__signal.append(float(0))


    def get_macd(self):
        return self.__macd

    def get_signal(self):
        return self.__signal


nvidia_macd = Macd(closing)

# Pierwszy wykres z MACD i SIGNAL
plt.figure(figsize=(12, 6))
plt.plot(dates[26:], nvidia_macd.get_macd()[26:], label='MACD', color='blue')
plt.plot(dates[26:], nvidia_macd.get_signal()[26:], label='SIGNAL', color="red")
plt.title('Stock MACD and Signal')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Drugi wykres z zamknięciami akcji NVIDIA
plt.figure(figsize=(12, 6))
plt.plot(dates[26:], closing[26:], label='NVIDIA STOCK', color="red")
plt.title('Stock Closing Values')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

