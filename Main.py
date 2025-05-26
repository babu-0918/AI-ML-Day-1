import pandas as pd
df = pd.read_csv('Titanic-Dataset.csv')
df = df.rename(coloumn={'Ticket':'Ticket ID'})
df.head()
df.isnull().any()
