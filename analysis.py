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
