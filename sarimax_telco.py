
import pandas as pd

from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

#Generuje zbiór uczący oraz testowy na bazie plików .csv i dziele dane na tygodnie
m_weekall = pd.read_csv('mcall.csv', usecols = ['time_int','month','day','hour','minutes','Weekday','SUMA'], parse_dates=['time_int'])
m_week = m_weekall [1440:7500]
m_weekall_test = pd.read_csv('mcalltest.csv', usecols = ['time_int','month','day','hour','minutes','Weekday','SUMA'], parse_dates=['time_int'])
m_week_test = m_weekall [1440:7500]

m_week.set_index('time_int', inplace=True)
#Przypisuje zbiory danych odpowiednio do zmiennych X i Y
X_train = m_week.iloc[:-1000,1:]
y_train = m_week.iloc[:-1000,0]

X_test = m_week_test.iloc[-1000:,1:]
y_test = m_week_test.iloc[-1000:,0]

y = m_week.iloc[:,0]
#Trenuje model algorytmem SARIMAX
model = SARIMAX(y,  order = (3, 1, 1),  seasonal_order =(1, 1, 0, 12))
result = model.fit()
result.summary()
start = len(X_train)
end = len(X_train) + len(X_test) - 1

predictions = result.predict(start, end, typ = 'levels').rename("Predictions")

