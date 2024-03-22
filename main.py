import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# wczytaj dane
df = pd.read_csv('data/wig20_d.csv')[-500:]

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


class Plots:
    def __init__(self, macd):
        self.__intersection_points = []
        self.__intersection_points_buy_or_sell = []
        self.macd_plot(macd)
        self.stock_plot()

    def stock_plot(self):
        dates_list = dates[26:].tolist()
        plt.figure(figsize=(20, 10))
        plt.plot(dates[26:], closing[26:], label='STOCK', color="red")
        plt.title('Stock Closing Values')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        for point in self.__intersection_points:
            plt.scatter(point, closing.array[dates_list.index(point)+26], color='green', zorder=5)
        plt.legend(['Value', 'BUY/SELL'])
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
        macd_values = macd.get_macd()[26:]
        signal_values = macd.get_signal()[26:]
        dates_list = dates[26:].tolist()  # Zamień DatetimeIndex na listę
        for i in range(1, len(macd_values)):
            if macd_values[i - 1] < signal_values[i - 1] and macd_values[i] > signal_values[i]:
                self.__intersection_points.append(dates_list[i])
                self.__intersection_points_buy_or_sell.append("buy")
            elif macd_values[i - 1] > signal_values[i - 1] and macd_values[i] < signal_values[i]:
                self.__intersection_points.append(dates_list[i])
                self.__intersection_points_buy_or_sell.append("sell")

        # Zaznacz punkty przecięcia na wykresie
        for point in self.__intersection_points:
            plt.scatter(point, macd_values[dates_list.index(point)], color='green', zorder=5 )

        plt.legend(['MACD', 'SIGNAL', 'BUY/SELL'])
        plt.show()

    def get_intersection_points(self):
        return self.__intersection_points

    def get_intersection_points_buy_or_sell(self):
        return self.__intersection_points_buy_or_sell


class Simulation:
    def __init__(self, plots):
        self.__stocks = 0
        self.__money = 100
        self.__points = plots.get_intersection_points()
        self.__account_balance = []
        self.__buy_or_sell = plots.get_intersection_points_buy_or_sell()
        self.simulate_trading()

    def simulate_trading(self):
        dates_list = dates[26:].tolist()
        for index, point in enumerate(self.__points):
            if self.__buy_or_sell[index] == "buy" and self.__money > 0:
                self.__stocks = self.__money / closing.array[dates_list.index(point)+26]
                self.__money = 0
            elif self.__buy_or_sell[index] == "sell" and self.__stocks > 0:
                self.__money = self.__stocks * closing.array[dates_list.index(point)+26]
                self.__stocks = 0
            # Obliczenie stanu konta dla bieżącego punktu
            current_balance = self.__money + self.__stocks * closing.array[dates_list.index(point)+26]
            self.__account_balance.append(current_balance)


    def get_final_balance(self):
        return self.__stocks, self.__money

    def plot_account_balance(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.__points, self.__account_balance, marker='o', linestyle='-')
        plt.title('Account Balance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Account Balance')
        plt.show()


ourMacd = Macd(closing)
myPlots = Plots(ourMacd)
simulation = Simulation(myPlots)
print(simulation.get_final_balance())
simulation.plot_account_balance()




