import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from IPython.core.display import display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
'exec(%matplotlib inline)'

dataset = pd.read_csv('Data/california_housing_test.csv')
display(dataset.describe())

dataset.isnull().any()
values = {'longitude': 0}
dataset = dataset.fillna(value=values)
display(dataset)

dataset = dataset.fillna(method='ffill')

Datacolumns = ["housing_median_age","latitude","total_rooms","total_bedrooms","population","households","longitude",
                "median_income"]

X = dataset[Datacolumns].values
y = dataset['median_house_value'].values

plt.figure(figsize=(10, 10))
plt.tight_layout()
seabornInstance.distplot(dataset['housing_median_age'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
accuracy = regressor.score(X_test,y_test)
print(accuracy*100,'%')

X = dataset.values
coeff_df = pd.DataFrame(regressor.coef_, Datacolumns, columns=['Coefficient'])
display(coeff_df)

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
dataDisplay = df.head(10)
display(dataDisplay)

dataDisplay.plot(kind='bar', figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.7', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.7', color='black')
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Accuracy:', accuracy*100,'%')