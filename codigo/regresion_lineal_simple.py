#Regresion Lineal Simple

import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_csv('/run/media/dantegrassi/Nuevo vol/inteligencia-artificial/data/Advertising.csv')

data = data.iloc[:, 1:]

print('''
    
    -----
    
    ''')

print(data.head())

print('''
    
    -----
    
    ''')

print(data.info())

print('''
    
    -----
    
    ''')

print(data.describe())

print('''
    
    -----
    
    ''')

print(data.columns)

print('''
    
    -----
    
    ''')

cols = ['TV', 'Radio', 'Newspaper']

for col in cols:
    plt.plot(data[col], data['Sales'], 'ro')
    plt.title('Ventas respecto a la publicidad en %s' %col)
    plt.show()