"""
Analysis code for titanic dataset.
"""
import pandas as pd

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

#####################
# Understand the data
print("The column names are %s" % (train_df.columns.values))
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

