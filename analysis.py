"""
Analysis code for titanic dataset.
"""

# Data wrangling
import pandas as pd

# Visualization
import seaborn as sea
import matplotlib.pyplot as plt

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

#####################
# Understand the data
print("The column names are {0}" % (train_df.columns.values))
# See first few rows.
train_df.head()
# Schema of the dataframe
train_df.info()
# Stats like means, etc for each column.
# Can also see missing data from 'count'
# Need 'all' so categorical columns like Name is included.
train_df.describe(include='all')

# For each feature, check distribution against label,
# to see if feature is promising.
# Each class, percent survival.
train_df[['Pclass', 'Survived']] \
    .groupby(['Pclass'], as_index=False) \
    .mean() \
    .sort_values(by='Survived', ascending=False)
# Each gender, percent survival.
train_df[['Sex', 'Survived']] \
    .groupby(['Sex'], as_index=False) \
    .mean() \
    .sort_values(by='Survived', ascending=False)

# Plot histogram
g = sea.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# Break histogram down with another dimension.
g = sea.FacetGrid(train_df, row='Pclass', col='Survived')
# Alpha = transparency
g.map(plt.hist, 'Age', alpha=.5, bins=20)

# Point plot. Show survival rate for [embarkations, pclass, gender]
# This means there's a chart / row per embarkation
g = sea.FacetGrid(train_df, row = 'Embarked')
# x = Pclass, y = Survived, breakdown = Sex
# Without palette, the color difference is not that striking
g.map(sea.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
# So the gender legend shows up
g.add_legend()

# Bar chart.
g = sea.FacetGrid(train_df, row='Embarked', col='Survived')
# If ci (confidence interval) exists, there is a vertical line on every bar.
g.map(sea.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

#####################
# Data wrangling


def impute_age(df):
    """
    Impute missing age values.
    @param df (DataFrame)
    @return (DataFrame) the modified dataframe.
    """
    
    # Impute age based on the median age in the [Sex, Pclass] group
    for sex in df['Sex'].unique():
        for pclass in df['Pclass'].unique():
            # The bitwise operator (instead of 'and') is actually required.
            # https://stackoverflow.com/a/36922103
            guess_age =  \
                df[(df['Sex'] == sex) & \
                   (df['Pclass'] == pclass)]['Age'].dropna().median()
            df.loc[df['Age'].isnull() & \
                   (df.Sex == sex) & \
                   (df.Pclass == pclass),\
                 'Age'] = guess_age
    return df

def update_features(df):
    """
    Drop, add, modify columns. To be applied on both training and test set.
    @param df (DataFrame)
    @return (DataFrame) the modified dataframe.

    """
    # Impute missing values
    df = impute_age(df)
    
    # Replace age with age group.
    df.loc[df['Age'] <= 16, 'AgeGroup'] = "kids (<=16)"
    df.loc[(df['Age'] > 16) & (df['Age'] <= 50), 'AgeGroup'] = "adults (>16,<= 50)"
    df.loc[df['Age'] > 50, 'AgeGroup'] = "elderly (>50)"
    
    # Add title column
    # See https://pandas.pydata.org/pandas-docs/stable/text.html
    # expand=False so it will return Index, not dataframe
    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
      'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    
    # Drop useless features
    df = df.drop(columns=['Ticket', 'Cabin', 'PassengerId', 'Name', 'Age'])
    return df

# Drop features
print("Before updateFeatures {0}".format(train_df.shape))
train_df = update_features(train_df)
print("After updateFeatures {0}".format(train_df.shape))
# Tally with group-by.
pd.crosstab(train_df['Title'], train_df['Sex'])
# To groupby-count one column: train_df.groupby('AgeGroup')['AgeGroup'].count()