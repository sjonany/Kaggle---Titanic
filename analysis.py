"""
Analysis code for titanic dataset.
"""

# Data wrangling
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sea

# Machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
KFOLD = 5

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
def impute_age(src_df, dst_df):
    """
    Impute missing age values.
    @param src_df (DataFrame). The data frame to gather statistics from.
    @return dst_df(DataFrame) The data frames to modify.
    """
    # Impute age based on the median age in the [Sex, Pclass] group
    for sex in src_df['Sex'].unique():
        for pclass in src_df['Pclass'].unique():
            # The bitwise operator (instead of 'and') is actually required.
            # https://stackoverflow.com/a/36922103
            guess_age =  \
                src_df[(src_df['Sex'] == sex) & \
                   (src_df['Pclass'] == pclass)]['Age'].dropna().median()
            dst_df.loc[dst_df['Age'].isnull() & \
               (dst_df.Sex == sex) & \
               (dst_df.Pclass == pclass),\
             'Age'] = guess_age

def impute_embarked(src_df, dst_df):
    """
    Impute missing embarkation values.
    @param src_df (DataFrame). The data frame to gather statistics from.
    @return dst_df (DataFrame) The data frames to modify.
    """
    freq_port = src_df.Embarked.dropna().mode()[0]
    dst_df['Embarked'].fillna(freq_port, inplace=True)

def impute_fare(src_df, dst_df):
    """
    Impute missing fare values.
    The train set is complete, but test set has 1 missing value.
    @param src_df (DataFrame). The data frame to gather statistics from.
    @return dst_df (DataFrame) The data frames to modify.
    """
    dst_df['Fare'].fillna(src_df['Fare'].dropna().median(), inplace=True)

def add_age_group(df):
    df.loc[df['Age'] <= 16, 'AgeGroup'] = "kids (<=16)"
    df.loc[(df['Age'] > 16) & (df['Age'] <= 50), 'AgeGroup'] = "adults (>16,<= 50)"
    df.loc[df['Age'] > 50, 'AgeGroup'] = "elderly (>50)"
    df['AgeGroup'] = df['AgeGroup'].astype('category')

def add_title(df):
    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
      'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].astype('category')

def add_is_alone(df):
    family_size_lst = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[family_size_lst == 1, 'IsAlone'] = 1
    
def update_features(src_df, dst_df):
    """
    Drop, add, modify columns. To be applied on both training and test set.
    @param src_df (DataFrame). The data frame to gather statistics from.
    @param dst_df (DataFrame) The data frames to modify.

    """
    # Impute missing values
    impute_age(src_df, dst_df)
    impute_embarked(src_df, dst_df)
    impute_fare(src_df, dst_df)
    
    # Add columns
    add_age_group(dst_df)
    add_title(dst_df)
    add_is_alone(dst_df)
    dst_df['Pclass'] = dst_df['Pclass'].astype('category')
    dst_df['Embarked'] = dst_df['Embarked'].astype('category')
    dst_df['Sex'] = dst_df['Sex'].astype('category')
    
    # Drop useless features
    dst_df.drop(columns=['Age', 'Cabin', 'Name', 'Parch',
                         'PassengerId', 'SibSp','Ticket'], inplace=True)
    numerize_categories(dst_df)

def numerize_categories(df):    
    """
    Convert all categorical columns to their codes.
    Some learning models hate strings.
    @param df (DataFrame).
    """
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

"""
Main
"""
raw_train_df = pd.read_csv(TRAIN_PATH)
raw_test_df = pd.read_csv(TEST_PATH)
train_df = raw_train_df.copy()
test_df = raw_test_df.copy()

update_features(raw_train_df, train_df)
update_features(raw_train_df, test_df)

train_features = train_df.drop(columns='Survived')
train_label = train_df["Survived"]

# Generate models
kfold = StratifiedKFold(n_splits=KFOLD)
models = {
        "SVM": SVC(),
        "Random forest": RandomForestClassifier(n_estimators=100)
        }

model_scores = {}
for model_name in models:
    total_score = 0
    for train_index, test_index in kfold.split(train_features, train_label):
        x_train = train_features.iloc[train_index,:]
        x_test = train_features.iloc[test_index,:]
        y_train = train_label.iloc[train_index]
        y_test = train_label.iloc[test_index]
        models[model_name].fit(x_train, y_train)
        total_score += models[model_name].score(x_test, y_test)
    model_scores[model_name] = total_score / KFOLD

desc_score_models = sorted(model_scores, key=model_scores.get, reverse=True)
for model in desc_score_models:
    print(model, model_scores[model])