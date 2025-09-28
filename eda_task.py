import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data load karo
df = pd.read_csv('titanic.csv')  # Yeh tumhari downloaded file path hoga

# Data preview
print("Pehli 5 rows:")
print(df.head())

# Data shape
print("\nRows aur Columns:")
print(df.shape)

# Missing values check
print("\nMissing values per column:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Histogram banao sab numeric columns ke liye
df.hist(figsize=(10, 8))
plt.suptitle('Histograms')
plt.show()

# Boxplot banao sab numeric columns ke liye
for col in df.select_dtypes(include=['float', 'int']):
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Correlation Matrix aur Heatmap
print("\nCorrelation Matrix:")
print(df.corr())
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot
sns.pairplot(df)
plt.suptitle('Pairplot')
plt.show()

# Outlier detection (example Fare column)
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Fare'] < Q1 - 1.5 * IQR) | (df['Fare'] > Q3 + 1.5 * IQR)]
print(f"\nFare mein outliers ki sankhya: {len(outliers)}")
