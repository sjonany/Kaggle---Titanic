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

#####################
# Analysis tools

def printDiscreteStats(df, colName):
    """
    Print tally for column, and see relationship against label.
    The label is hardcoded to 'Survived'.
    @param df (dataframe)
    @param colName (string) - a feature column in 'df'.
    """
    print(df[[colName, 'Survived']] \
        .groupby([colName]) \
        .agg(['mean', 'count']) \
        .sort_values(
                by=[('Survived', 'mean')],
                ascending=False))

#####################
# Understand the data

def printPrelimAnalysis(df):
    """
    Preliminary analysis. Just visualizations before we wrangle anything.
    @param df (dataframe)
    """
    print("The column names are {0}" % (df.columns.values))
    # See first few rows.
    df.head()
    # Schema of the dataframe
    df.info()
    # Stats like means, etc for each column.
    # Can also see missing data from 'count'
    # Need 'all' so categorical columns like Name is included.
    df.describe(include='all')
    
    # For each feature, check distribution against label,
    # to see if feature is promising.
    # Each class, percent survival.
    df[['Pclass', 'Survived']] \
        .groupby(['Pclass'], as_index=False) \
        .mean() \
        .sort_values(by='Survived', ascending=False)
    # Each gender, percent survival.
    df[['Sex', 'Survived']] \
        .groupby(['Sex'], as_index=False) \
        .mean() \
        .sort_values(by='Survived', ascending=False)
    
    # Plot histogram
    g = sea.FacetGrid(df, col='Survived')
    g.map(plt.hist, 'Age', bins=20)
    
    # Break histogram down with another dimension.
    g = sea.FacetGrid(df, row='Pclass', col='Survived')
    # Alpha = transparency
    g.map(plt.hist, 'Age', alpha=.5, bins=20)
    
    # Point plot. Show survival rate for [embarkations, pclass, gender]
    # This means there's a chart / row per embarkation
    g = sea.FacetGrid(df, row = 'Embarked')
    # x = Pclass, y = Survived, breakdown = Sex
    # Without palette, the color difference is not that striking
    g.map(sea.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
    # So the gender legend shows up
    g.add_legend()
    
    # Bar chart.
    g = sea.FacetGrid(df, row='Embarked', col='Survived')
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

def impute_embarked(df):
    freq_port = df.Embarked.dropna().mode()[0]
    df['Embarked'] = df['Embarked'].fillna(freq_port)
    return df
    
def update_features(df):
    """
    Drop, add, modify columns. To be applied on both training and test set.
    @param df (DataFrame)
    @return (DataFrame) the modified dataframe.

    """
    # Impute missing values
    df = impute_age(df)
    df = impute_embarked(df)
    
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
    
    # Add IsAlone as a feature
    family_size_lst = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[family_size_lst == 1, 'IsAlone'] = 1
    
    # Drop useless features
    df = df.drop(columns=['Age', 'Cabin', 'Name', 'Parch', 'PassengerId',
                          'SibSp','Ticket'])
    return df

"""
Main
"""
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print("Before updateFeatures {0}".format(train_df.shape))
train_df = update_features(train_df)
print("After updateFeatures {0}".format(train_df.shape))
