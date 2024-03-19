import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# wczytaj dane
df = pd.read_csv('data/btc_v_d.csv')[-1000:]

# tworzenie zmiennych ktore przechwouja daty i wartosci zamkniecia danego dnia
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


class Plot:
    def __init__(self, macd=None):
        if macd is None:
            self.stock_plot()
        else:
            self.macd_plot(macd)

    def stock_plot(self):
        plt.figure(figsize=(12, 6))
        plt.plot(dates[26:], closing[26:], label='STOCK', color="red")
        plt.title('Stock Closing Values')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    def macd_plot(self, macd):
        # Pierwszy wykres z MACD i SIGNAL z zaznaczonymi punktami przecięcia
        plt.figure(figsize=(20, 10))
        plt.plot(dates[26:], macd.get_macd()[26:], label='MACD', color='blue')
        plt.plot(dates[26:], macd.get_signal()[26:], label='SIGNAL', color="red")
        plt.title('Stock MACD and Signal with Intersection Points')
        plt.xlabel('Date')
        plt.ylabel('Value')

        plt.legend()

        # Znajdź punkty przecięcia
        intersection_points = []
        macd_values = macd.get_macd()[26:]
        signal_values = macd.get_signal()[26:]
        dates_list = dates[26:].tolist()  # Zamień DatetimeIndex na listę
        for i in range(1, len(macd_values)):
            if (macd_values[i - 1] < signal_values[i - 1] and macd_values[i] > signal_values[i]) or \
                    (macd_values[i - 1] > signal_values[i - 1] and macd_values[i] < signal_values[i]):
                intersection_points.append(dates_list[i])  # Indeksujemy listę zamiast DatetimeIndex

        # Zaznacz punkty przecięcia na wykresie
        for point in intersection_points:
            plt.scatter(point, macd_values[dates_list.index(point)], color='green', zorder=5 )

        plt.legend(['MACD', 'SIGNAL', 'BUY/SELL'])
        plt.show()


ourMacd = Macd(closing)
stockPlot = Plot()
macdPlot = Plot(ourMacd)





