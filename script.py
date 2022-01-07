# Import libraries
import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Load into a dataframe
df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

# Inspect dataframe
print(df.head())

# Group df by year column and calculate the mean of totalprod
prod_per_year = df.groupby('year').totalprod.mean().reset_index()

# Select year column as in prod_per_year
X = prod_per_year.year

# Reshape (Rotate) X
X = X.values.reshape(-1, 1)

# Select totalprod column in prod_per_year
y = prod_per_year.totalprod

# create scatter plot
plt.scatter(X, y)

# Create a linear regression model
regr = linear_model.LinearRegression()

# Fit model to the data
regr.fit(X, y)

# Print out the slope and the intercept of the line
print('Slope = {}\nIntercept = {}'.format(regr.coef_[0], regr.intercept_))

#  Get the predictions from the regr object
y_predict = regr.predict(X)

# create a line plot
plt.plot(X, y_predict)

# Show plots
plt.show()

# Create a Numpy array
X_future = np.array(range(2013, 2050))

# Rotate X_future
X_future = X_future.reshape(-1, 1)

# Get the y-values that the model would predict on X_future
future_predict = regr.predict(X_future)

# Clear current plot
plt.clf()

# Plot a line graph
plt.plot(X_future, future_predict)

# Show plot
plt.show()