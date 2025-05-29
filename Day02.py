import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
sns.set=(colour_code=true)

df = pd.read_csv('Titanic-Database.csv')

#cheeck Summary statistics

print(df.describe())

#for headings
df.head()

#for data information
df.info()

#plot bar graphs between two coloumns
sns.barplot(df['Survived'],df['Age'])

#for single coloumn distribution

sns.distplot(df['Age'])

# Histogram

plt.hist(df["Age"], bins=30)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Column Name")
plt.show()

# Scatter Plot
sns.scatterplot(x=df["Age"], y=df["Cabin"])
plt.title("Scatter Plot")
plt.show()

print(df.corr())  # Displays correlation matrix

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

group1 = df[df["Age"] == "A"]["36"]
group2 = df[df["PassengerId"] == "B"]["2"]

t_stat, p_value = stats.ttest_ind(group1, group2)
print("T-statistic:", t_stat)
print("P-value:", p_value)

