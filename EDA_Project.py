# import the libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# read the dataset
df = pd.read_csv('insurance.csv')

# separating the numerical and categorical feature
# + data quality checks:
num_cols = ['age', 'bmi', 'children', 'charges']
cat_cols = ['sex', 'smoker', 'region']


df[num_cols].nunique()
df.describe(include='object')

df.describe()

df.info()
df.isnull().sum()
# in some cases that you run the above code and you will conculde that there is no
# Null values, sometimes datasets go crazy and they way put some crazy number to suggest
# that there is a Null value in place like for age they put smth like 999 or -999 or "unkown"
# you should always double check with value_count to make sure there is no inputation for
# missing or null values.

for col in df.columns:
    print(df[col].value_counts())
    print('-'*50)
    
    

    
    
df[num_cols].mean()


df.duplicated().sum()
dup_mask = df.duplicated()
df[dup_mask]
df = df.drop_duplicates()
df.dtypes

# univariate analysis
df.describe()

df[num_cols].skew()


df[num_cols].kurtosis()


for col in num_cols:
    plt.figure(figsize=(10,7))
    sns.histplot(x = df[col], kde=True)
    plt.xlabel(f'{col}')
    plt.ylabel('frequency')
    
for col in num_cols:
    plt.figure(figsize=(10,7))
    sns.boxplot(x = df[col])
    plt.xlabel(f'{col}')
    plt.ylabel('frequency')
    

for cat in cat_cols:
    print(f'frequency table for {cat}')
    print(df[cat].value_counts())
    print(f'proportion table for {cat}')
    print(df[cat].value_counts(normalize=True))
    print('-'*50)
    # in tree based algorithms, it can handle the outlier but in Linear models
    # we need to handle them.      

   
for col in cat_cols:
    plt.figure(figsize=(10,7))
    sns.countplot(x = df[col])
    plt.xlabel(f'{col}')
    plt.ylabel('frequency')
    
    
    # the issue of the data being inbalanced in a feature is still a problem but in the case
    # of TARGET variable that is and critical and huge red flag.
    
    
# Bivariate analysis
# Numerical vs numerical features

num_pairs = {
    ('age' , 'charges'),
    ('bmi', 'charges'),
    ('children' , 'charges'),
}

for x, y in num_pairs:
    plt.figure(figsize=(10,7))
    sns.scatterplot(x = df[x] , y = df[y])
  
    
# categorical vs categorical
    
print(pd.crosstab(df['sex'] , df['smoker']))
sns.heatmap(df[num_cols].corr() , annot=True , cmap='coolwarm')


# making the charges categorical
df['charges'].describe()



conditions = [
    df['charges'] > 9400,
    df['charges'] < 4800,
    (df['charges'] >= 4800) & (df['charges'] <= 9400)
]

# Define the corresponding choices for each condition
choices = ['high', 'low', 'medium']

# Create the 'charges_cat' column
df['charges_cat'] = np.select(conditions, choices, default='unknown')

# Display the first few rows to verify
print(df.head())



print(pd.crosstab(df['sex'] , df['charges_cat']))
print(pd.crosstab(df['smoker    '] , df['charges_cat']))



# Numerical vs categrical analysis

num_for_box = ['age' , 'charges','bmi']
cat_for_box = ['sex' , 'smoker' , 'region']


for num in num_for_box:
    for cat in cat_for_box:
        plt.figure(figsize=(7,5))
        sns.boxplot(x = df[cat], y = df[num])





df_num = df[num_cols]
df_num.groupby(df['smoker']).mean()
df_num.groupby(df['sex']).mean()


for cat in ['sex', 'smoker','region']:
    print(f'group by {cat}')
    print(df_num.groupby(df[cat]).mean())
    print('-'*50)

    # one intersting point is that the smoker
    # will be charged more in their charges.
    
    
    
    
    

df['region'].unique()
df['charges'].groupby(df['region']).mean()

r , p = pearsonr(df['bmi'] , df['charges'])



# multi-variate analysis
pairplot_cols = num_cols + ['smoker']
df_pair = df[pairplot_cols].dropna()

# make pair plot
plt.figure(figsize=(10,7))
sns.pairplot(df_pair, hue='smoker')
plt.show()


sns.clustermap(df[num_cols].corr() , annot = True )



# Outlier detection
def iqr_bounds(series, factor=1.5):
        q1 = series.quantile(0.25)
        q2 = series.quantile(0.75)
        iqr = q2 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q2 + factor * iqr

        return lower_bound, upper_bound, iqr

iqr_bounds(df['bmi'])

for col in num_cols:
    col_clean = df_num[col].dropna()
    lower , upper , iqr = iqr_bounds(col_clean)
    outlier_mask = (col_clean > upper) | (col_clean < lower)
    num_outlier = outlier_mask.sum()
    total = col_clean.shape[0]
    
    print(f'IQR analysis for {col}')
    print(f'lower bound {lower:.2f}')
    print(f'Upper bound {upper:.2f}')
    print(f'IQR: {iqr}')
    print(f'we have {num_outlier} outliers out of {total}')
    print('-'*50)




z_threshold = 3.0
for col in num_cols:
    col_clean = df_num[col].dropna()
    mean = col_clean.mean()
    std = col_clean.std()
    z_score = (col_clean - mean) / std
    outlier_mask = np.abs(z_score) > z_threshold
    num_outlier = outlier_mask.sum()
    total = col_clean.shape[0] 

    print(f'IQR analysis for {col}')
    print(f'mean: {mean} , std: {std}')
    print(f'threshold: {z_threshold}')
    print(f'outlier {num_outlier} in {total}')
    print('-'*50)
    
# the important numerical features that is correlated with charges are sex and bmi and 
# the number of children do not effect the charges much to consider it for ML feature.

