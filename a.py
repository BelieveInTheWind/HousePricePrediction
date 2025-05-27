import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("AmesHousing.csv")
dataset.head()
dataset.info
#Dealing with outliers
pd.DataFrame([dataset.corr(numeric_only=True)['SalePrice'].sort_values()])

sns.scatterplot(data=dataset, x='Overall Qual', y='SalePrice')
plt.axhline(y=200000,color='r')
dataset[(dataset['Overall Qual']>8) &(dataset['SalePrice']<200000)][['SalePrice', 'Overall Qual']]
sns.scatterplot(x='Gr Liv Area', y='SalePrice', data=dataset)
plt.axhline(y=200000, color='r')
plt.axvline(x=4000, color='r')
dataset[(dataset['Gr Liv Area']>4000) & (dataset['SalePrice']<400000)][['SalePrice', 'Gr Liv Area']]
dataset_total =dataset
index_drop=dataset[(dataset['Gr Liv Area']>4000) & (dataset['SalePrice']<400000)].index
dataset=dataset.drop(index_drop, axis=0)
sns.scatterplot(x='Gr Liv Area', y='SalePrice', data=dataset_total, color='red')

sns.scatterplot(x='Gr Liv Area', y='SalePrice', data=dataset, color='green')
plt.axhline(y=200000, color='m')
plt.axvline(x=4000, color='m')
sns.scatterplot(x='Overall Qual', y='SalePrice', data=dataset)
plt.axhline(y=200000,color='r')
sns.boxplot(x='Overall Qual', y='SalePrice', data=dataset)
#Dealing with missing data
dataset.head()
dataset= dataset.drop('PID', axis=1)
dataset.head()
dataset.isnull()
dataset.isnull().sum()
100 * (dataset.isnull().sum()/len(dataset))
def missing_percent(df):
    nan_percent= 100*(df.isnull().sum()/len(df))
    nan_percent= nan_percent[nan_percent>0].sort_values()
    return nan_percent
nan_percent= missing_percent(dataset)

nan_percent
plt.figure(figsize=(12,6))
sns.barplot(x=nan_percent.index, y=nan_percent)
plt.xticks(rotation=90)
plt.figure(figsize=(12,6))
sns.barplot(x=nan_percent.index, y=nan_percent)
plt.xticks(rotation=90)

#Set 1% threshold:
plt.ylim(0,1)
nan_percent[nan_percent < 1]

nan_percent[nan_percent<1].index

100/len(dataset)
dataset[dataset['Electrical'].isnull()]
dataset[dataset['Garage Area'].isnull()]
dataset= dataset.dropna(axis=0, subset=['Electrical', 'Garage Area'])

nan_percent= missing_percent(dataset)

plt.figure(figsize=(12,6))
sns.barplot(x=nan_percent.index, y=nan_percent)
plt.xticks(rotation=90)
plt.ylim(0,1)
dataset[dataset['Total Bsmt SF'].isnull()]
dataset[dataset['Bsmt Half Bath'].isnull()]

dataset[dataset['Bsmt Full Bath'].isnull()]

#Numerical Columns fill with 0:
bsmt_num_cols= ['BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF','Total Bsmt SF' ,'Bsmt Full Bath', 'Bsmt Half Bath']
dataset[bsmt_num_cols]=dataset[bsmt_num_cols].fillna(0)

#String Columns fill with None:
bsmt_str_cols= ['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']
dataset[bsmt_str_cols]= dataset[bsmt_str_cols].fillna('None')
nan_percent= missing_percent(dataset)

plt.figure(figsize=(12,6))
sns.barplot(x=nan_percent.index, y=nan_percent)
plt.xticks(rotation=90)
plt.ylim(0,1)
dataset["Mas Vnr Type"]= dataset["Mas Vnr Type"].fillna("None")
dataset["Mas Vnr Area"]= dataset["Mas Vnr Area"].fillna(0)
nan_percent= missing_percent(dataset)

plt.figure(figsize=(12,6))
sns.barplot(x=nan_percent.index, y=nan_percent)
plt.xticks(rotation=90)
dataset[['Garage Type', 'Garage Yr Blt', 'Garage Finish', 'Garage Qual', 'Garage Cond']]
#Filling the missing Value:
Gar_str_cols= ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']
dataset[Gar_str_cols]=dataset[Gar_str_cols].fillna('None')

dataset['Garage Yr Blt']=dataset['Garage Yr Blt'].fillna(0)
nan_percent= missing_percent(dataset)

plt.figure(figsize=(12,6))
sns.barplot(x=nan_percent.index, y=nan_percent)
plt.xticks(rotation=90)
nan_percent.index

dataset[['Lot Frontage', 'Fireplace Qu', 'Fence', 'Alley', 'Misc Feature',
       'Pool QC']]
dataset= dataset.drop(['Fence', 'Alley', 'Misc Feature','Pool QC'], axis=1)

nan_percent= missing_percent(dataset)

plt.figure(figsize=(12,6))
sns.barplot(x=nan_percent.index, y=nan_percent)
plt.xticks(rotation=90)
dataset['Neighborhood'].unique()
plt.figure(figsize=(8,12))
sns.boxplot(data=dataset, x='Lot Frontage', y='Neighborhood')
dataset.groupby('Neighborhood')['Lot Frontage']
dataset.groupby('Neighborhood')['Lot Frontage'].mean()
dataset.groupby('Neighborhood')['Lot Frontage'].transform(lambda val: val.fillna(val.mean()))
dataset['Lot Frontage']=dataset.groupby('Neighborhood')['Lot Frontage'].transform(lambda val: val.fillna(val.mean()))
nan_percent= missing_percent(dataset)

plt.figure(figsize=(12,6))
sns.barplot(x=nan_percent.index, y=nan_percent)
plt.xticks(rotation=90)
dataset['Lot Frontage']= dataset['Lot Frontage'].fillna(0)
nan_percent= missing_percent(dataset)
nan_percent


dataset['MS SubClass']
dataset.info()
dataset['MS SubClass'].unique()
dataset['MS SubClass'] = dataset['MS SubClass'].apply(str)
dataset['MS SubClass']
dataset.select_dtypes(include='object')
dataset_num = dataset.select_dtypes(exclude='object')
dataset_obj = dataset.select_dtypes(include='object')
dataset_num.info()
dataset_obj.info()
dataset_obj = pd.get_dummies(dataset_obj, drop_first=True)
dataset_obj.shape
dataset_num.head()
finaldata = pd.concat([dataset_num, dataset_obj], axis=1)
finaldata.head()