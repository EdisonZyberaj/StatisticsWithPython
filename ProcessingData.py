# Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset
dataset = pd.read_csv("pima-indians-diabetes.csv")

# Identify missing data (assumes missing data is represented as NaN)
missing_values = dataset.isnull().sum()

# Print the number of missing entries in each column
print(missing_values)

# Separate features (X) from the target (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Configure an instance of the SimpleImputer class (replace missing values with mean)
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

# Fit the imputer on the numerical feature matrix
imputer.fit(X)

# Apply the transform to replace missing values
X = imputer.transform(X)

# Print your updated matrix of features
print(X)
