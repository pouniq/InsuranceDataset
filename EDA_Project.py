# import the libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# read the dataset
df = pd.read_csv('insurance.csv')

# separating the numerical and categorical feature
num_cols = ['age', 'bmi', 'children', 'charges']
cat_cols = ['sex', 'smoker', 'region']


# univariate analysis
df[num_cols].nunique()
df.describe(include='object')

df.describe()

df.info()

df[num_cols].mean()

df.duplicated().sum()
df = df.drop_duplicates()
df.dtypes

# Bivariate analysis


# Numerical vs numerical features
for col in num_cols:
    plt.figure(figsize=(10,7))
    sns.histplot(x = df[col])
    plt.xlabel(f'{col}')
    plt.ylabel('frequency')
    
for col in num_cols:
    plt.figure(figsize=(10,7))
    sns.boxplot(x = df[col])
    plt.xlabel(f'{col}')
    plt.ylabel('frequency')
    
# categorical vs categorical
    
for cat in cat_cols:
    print(f'frequency table for {cat}')
    print(df[cat].value_counts())
    print(f'proportion table for {cat}')
    print(df[cat].value_counts(normalize=True))
    print('-'*50)

   
for col in cat_cols:
    plt.figure(figsize=(10,7))
    sns.countplot(x = df[col])
    plt.xlabel(f'{col}')
    plt.ylabel('frequency')


for col in cat_cols:
    plt.figure(figsize=(10,7))
    sns.countplot(x = df[col])
    plt.xlabel(f'{col}')
    plt.ylabel('frequency')
  

print(pd.crosstab(df['sex'] , df['smoker']))
sns.heatmap(df[num_cols].corr() , annot=True , cmap='coolwarm')
# Numerical vs categrical analysis

num_pairs = {
    ('age' , 'charges'),
    ('bmi', 'charges'),
    ('children' , 'charges'),
}

for x, y in num_pairs:
    plt.figure(figsize=(10,7))
    sns.scatterplot(x = df[x] , y = df[y])


df_num = df[num_cols]
df_num.groupby(df['smoker']).mean()
df_num.groupby(df['sex']).mean()


df['region'].unique()
df['charges'].groupby(df['region']).mean()

r , p = pearsonr(df['bmi'] , df['charges'])