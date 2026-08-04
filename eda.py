import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/titanic.csv")

print("=" * 50)
print("First 5 Rows")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Information")
print(df.info())

print("\nSummary Statistics")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())

if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

if "Cabin" in df.columns:
    df = df.drop(columns=["Cabin"])

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# Survival Count
plt.figure(figsize=(6,4))
df["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Passengers")
plt.tight_layout()
plt.show()

# Gender Distribution
plt.figure(figsize=(6,4))
df["Sex"].value_counts().plot(kind="bar")
plt.title("Gender Distribution")
plt.tight_layout()
plt.show()

# Age Distribution
plt.figure(figsize=(8,5))
plt.hist(df["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Passenger Class
plt.figure(figsize=(6,4))
df["Pclass"].value_counts().sort_index().plot(kind="bar")
plt.title("Passenger Class")
plt.tight_layout()
plt.show()

# Survival Rate by Gender
survival_by_gender = df.groupby("Sex")["Survived"].mean()

plt.figure(figsize=(6,4))
survival_by_gender.plot(kind="bar")
plt.title("Survival Rate by Gender")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()

# Correlation Matrix
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(10,8))
plt.imshow(numeric_df.corr(), cmap="coolwarm", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(numeric_df.columns)), numeric_df.columns, rotation=90)
plt.yticks(range(len(numeric_df.columns)), numeric_df.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

print("\nAverage Age:", round(df["Age"].mean(), 2))
print("Median Fare:", round(df["Fare"].median(), 2))
print("Overall Survival Rate:", round(df["Survived"].mean() * 100, 2), "%")
