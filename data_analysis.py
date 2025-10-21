# Suppress future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Import libraries
import pandas as pd
# Load dataset
df = pd.read_csv('titanic.csv')
# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())
# Dataset info
print("\nDataset Info:")
print(df.info())
# Basic statistics
print("\nBasic Statistics:")
print(df.describe())
# Handle missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df.dropna(subset=['Embarked'], inplace=True)
# Check missing values after cleaning
print("\nMissing Values after Cleaning:")
print(df.isnull().sum())
# Analyze survival counts
print("\nSurvival Count:")
print(df['Survived'].value_counts())
# Survival rate by passenger class
print("\nSurvival Rate by Passenger Class:")
print(df.groupby('Pclass')['Survived'].mean())
# Survival rate by gender
print("\nSurvival Rate by Gender:")
print(df.groupby('Sex')['Survived'].mean())
#Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Count plot of survival
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()
# Survival rate by Passenger Class
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.show()
# Survival rate by Gender
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.show()
