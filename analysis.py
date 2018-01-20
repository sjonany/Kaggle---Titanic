"""
Analysis code for titanic dataset.
"""

# Data wrangling
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sea
import sys

# Machine learning
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
KFOLD = 10
RANDOM_STATE = 123

#####################
# Analysis tools

def print_discrete_stats(df, colName):
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

def print_hist(arr):
    """
    Draw histogram to show distribution of arr
    @param arr (1D Dataframe) - an array of real numbers
    """
    plt.hist(arr, bins=20)
    plt.axvline(arr.mean(), color='b', linestyle='dashed', linewidth=2)

#####################
# Understand the data
def print_prelim_analysis(df):
    """
    Preliminary analysis. Just visualizations before we wrangle anything.
    @param df (dataframe)
    """
    print("The column names are {0}".format(df.columns.values))
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

def plot_variable_importance( X , y ):
	"""
	Plot variable importance by training decision tree.
	@param X (DataFrame). The feature set.
	@param y (DataFrame). The labels.
	"""
    model = DecisionTreeClassifier()
    model.fit( X , y )
    plot_model_variable_importance( X, model )
    
    print ("Tree model score on training set: {0}".format(
            model.score( X , y )))

def plot_model_variable_importance(X, model, title=""):
	"""
	Plot variable importance of a trained model.
	@param X (DataFrame). The feature set
	@param model (Model). The trained model
	@param title (string). Title of the plot
	"""
    if not hasattr(model, 'feature_importances_'):
        return
    feature_importances = model.feature_importances_
    max_num_features = 20
    imp = pd.DataFrame( 
        feature_importances, 
        columns = [ 'Importance' ] ,
        index = X.columns
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : max_num_features ].plot( kind='barh', title=title )
    
def plot_pearson(df):
    """
    Plot pearson correlation for a dataframe.
    Useful to see if all the any of the features are redundant.
    Or in general, another way to sanity check the features.
    @param df (DataFrame) The entire training dataframe (including label).
    """
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14,12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sea.heatmap(df.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)

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
            for title in src_df['Title'].unique():
                # The bitwise operator (instead of 'and') is actually required.
                # https://stackoverflow.com/a/36922103
                guess_age =  \
                    src_df[(src_df['Sex'] == sex) & \
                       (src_df['Pclass'] == pclass) & \
                       (src_df['Title'] == title)]['Age'].dropna().median()
                dst_df.loc[dst_df['Age'].isnull() & \
                   (dst_df['Sex'] == sex) & \
                   (dst_df['Pclass'] == pclass) & \
                   (dst_df['Title'] == title),\
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
    bins = (-1, 16, 50, 100)
    age_groups = ["kids_le16", "adults_g16_le50)", "elderly_g50"]
    age_group_col = pd.cut(df.Age, bins, labels=age_groups)
    df['AgeGroup'] = age_group_col.astype('category')

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
    
def add_has_cabin(df):
    df['HasCabin'] = df['Cabin'].apply(lambda x: type(x) == float) 
    
def update_features(src_df, dst_df):
    """
    Drop, add, modify columns. To be applied on both training and test set.
    This form is not dependent on the learning model, and is also a friendly
    format to do plots and analyses on, as the enums are in legible forms.
    So, more processing like 'onehot_categories' might be needed.
    @param src_df (DataFrame). The data frame to gather statistics from.
    @param dst_df (DataFrame) The data frames to modify.
    @return dst_df The updated dst_df.
    Side-effect - Will modify dst_df, but you have to reassign with the return
    value. Idk how to select columns and mutate dst_df :/
    """
    # Title needed for age imputation
    add_title(src_df)
    add_title(dst_df)
    impute_age(src_df, src_df)
    impute_age(src_df, dst_df)
    add_age_group(dst_df)

    impute_embarked(src_df, dst_df)
    impute_fare(src_df, dst_df)    
    add_is_alone(dst_df)
    add_has_cabin(dst_df)
    
    dst_df['Pclass'] = dst_df['Pclass'].astype('category')
    dst_df['Embarked'] = dst_df['Embarked'].astype('category')
    dst_df['Sex'] = dst_df['Sex'].astype('category')
    
    # Scaling
    features_to_scale = ['Age', 'Fare']
    scaler = preprocessing.StandardScaler().fit(src_df[features_to_scale])
    dst_df[features_to_scale] = scaler.transform(dst_df[features_to_scale])
    
    # Select features
    dst_df = dst_df[[
            'Age',
            'AgeGroup',
            'Embarked',
            'Fare',
            'HasCabin',
            'IsAlone',
            'Pclass',
            'Sex',
            'Title'
            ]].copy()
    return dst_df

def numerize_categories(df):    
    """
    Convert all categorical columns to their codes.
    Some learning models hate strings.
    Note that some learning models do not handle enums, and you might have to
    use 'onehot_categories()' instead.
    @param df (DataFrame).
    """
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    
def onehot_categories(df):    
    """
    Apply one-hot encoding to all categorical columns.
    Some sklearn models don't deal with categories.
    E.g. As of Jan 2018, even random forest implementation converts enums
      to floats. https://github.com/scikit-learn/scikit-learn/pull/4899
    @param df (DataFrame).
    @return df new pd with categorical columns replaced with binaries.
    """
    cat_cols = df.select_dtypes(['category']).columns
    # drop_first to reduce number of dependent features. The last item can be
    # inferred. We could do better by maybe not just dropping arbitrarily.
    # E.g. keep the one that has the highest variable importance? Or drop the
    # one with highest occurrence.
    dummied_cat_df = pd.get_dummies(df[cat_cols], drop_first = True)
    dummied_pd = pd.concat([df, dummied_cat_df], axis=1)
    dummied_pd.drop(cat_cols, axis=1, inplace=True)
    return dummied_pd

def get_selective_features(features, labels):
    """
    Get feature names that are important.
    Importance is evaluated by fitting tree to the training set.
    @param features (DataFrame)
    @param labels (DataFrame)
    @return list<string> the feature names
    """
    clf = ExtraTreesClassifier(random_state=RANDOM_STATE)
    clf = clf.fit(features, labels)
    feature_select_model = SelectFromModel(clf, prefit=True)
    return features.columns[feature_select_model.get_support()]

def get_ensemble_feat(model, x_train, y_train, x_test, nfold):
    """
    Generate 1 column that contains predictions made by model.
    The model is trained on folds that don't contain the data point to predict
    on.
    @param model (Model)
    @param x_train (Dataframe)
    @param y_train (Dataframe)
    @param x_test (Dataframe)
    @param nfold (int) number of folds for generating train cols.
    @return (1D Dataframe, 1D Dataframe) - the columns to add to x_train and
      x_test respectively.
    """
    ensemble_train_col = pd.Series(np.zeros(x_train.shape[0]))
    kfold = StratifiedKFold(n_splits=nfold, random_state=RANDOM_STATE)
    for train_indices, test_indices in kfold.split(x_train, y_train):
        x_train_fold = x_train.iloc[train_indices,:]
        y_train_fold = y_train.iloc[train_indices]
        x_test_fold = x_train.iloc[test_indices,:]
        model.fit(x_train_fold, y_train_fold)
        ensemble_train_col[test_indices] = model.predict(x_test_fold)
    model.fit(x_train, y_train)
    ensemble_test_col = model.predict(x_test)
    return (ensemble_train_col, ensemble_test_col)

def get_ensemble_feats(models, x_train, y_train, x_test, nfold):
    """
    Apply get_ensemble_feat() to all the models to obtain the second-layer
    feature sets.
    @param models (Map<String, Model>) First-layer models
    @return (Dataframe, Dataframe) - the ensemble x_train and x_test
    """
    # Map <String model, list<prediction>>
    ensemble_train_feats = {}
    ensemble_test_feats = {}
    for model_name in models:
        x_train_col, x_test_col = get_ensemble_feat(
                models[model_name], x_train, y_train, x_test, nfold)
        ensemble_train_feats[model_name] = x_train_col
        ensemble_test_feats[model_name] = x_test_col
    return pd.DataFrame(data=ensemble_train_feats), \
           pd.DataFrame(data=ensemble_test_feats)
        

#####################
# Model generation and evalution

def evaluate_models(models, nfold, features, labels):
    """
    Perform k-fold on models, 
    Print K-fold accuracy for models.
    @param nfold (int) k in kfold
    @param models (Map<string, model>) Models to evaluate.
    @param features, labels. X,Y of training set.
    """
    model_scores = {}
    kfold = StratifiedKFold(n_splits=nfold, random_state = RANDOM_STATE)
    for model_name in models:
        total_score = 0
        for train_index, test_index in kfold.split(features, labels):
            x_train = features.iloc[train_index,:]
            x_test = features.iloc[test_index,:]
            y_train = labels.iloc[train_index]
            y_test = labels.iloc[test_index]
            models[model_name].fit(x_train, y_train)
            cur_score = models[model_name].score(x_test, y_test)
            total_score += cur_score
            print("Model {0}. Acc {1}".format(model_name, cur_score))
        model_scores[model_name] = total_score / nfold
        plot_model_variable_importance(features, models[model_name],
                                       title=model_name)
    
    desc_score_models = sorted(model_scores, key=model_scores.get, reverse=True)
    for model in desc_score_models:
        print(model, model_scores[model])
        
def gen_models():
    """
    Create untrained models
    
    @return (Map<string, model>) Models to evaluate.
    """
    models = {
        "SVM": SVC(random_state=RANDOM_STATE),
        # See grid_search_forest()
        "Random forest": RandomForestClassifier(n_estimators=50,
                                                max_features=2,
                                                random_state=RANDOM_STATE),
        # See grid_search_xgboost()                                                
        "Xgboost": xgb.XGBClassifier(
                colsample_bytree = 0.75,
                subsample = 0.5,
                max_depth = 10,
                n_estimators = 1000,
                learning_rate = 0.01,
                seed=RANDOM_STATE),
        }
    return models

def gen_second_layer_model():
    """
    Create the second layer model that will be trained on first-layer model's
    outputs.
    
    @return Model
    """
    # Tuned using grid_search_xgboost
    return xgb.XGBClassifier(
                colsample_bytree = 0.75,
                subsample = 0.5,
                max_depth = 1,
                n_estimators = 100,
                learning_rate = 0.01,
                seed=RANDOM_STATE)

def grid_search_xgboost(features, labels):
    """
    Grid search on xgboost.
    Jan 18, 2018
    Out: Best parameters set found on development set:
    {'colsample_bytree': 0.75, 'learning_rate': 0.01, 'max_depth': 10,
     'n_estimators': 1000, 'subsample': 0.5}
    
    Out: Best params on ensemble set:
    {'colsample_bytree': 0.75, 'learning_rate': 0.01, 'max_depth': 1, 
    n_estimators': 100, 'subsample': 0.5}        
        
    @param features, labels. X,Y of training set.
    """
    boost_params = {'n_estimators': [100,500], # [100, 500, 1000]
                    'learning_rate': [0.1, 0.01], # [0.1, 0.01, 0.001]
                    'subsample': [0.5, 0.75, 1.0], # [0.5, 0.75, 1.0]
                    'colsample_bytree': [0.75], # [0.5, 0.75, 1.0],
                    # I think this is 'min leaf weight', but Idk how to tune.
                    # 'min_child_weight': 
                    'max_depth': [1, 2, 3]}# [4, 6, 8, 10]}
    cv_model = GridSearchCV(\
                xgb.XGBClassifier( \
                        seed=RANDOM_STATE), boost_params, cv=5,\
                       scoring='accuracy', verbose=10)
    cv_model.fit(features, labels)
    print("Best parameters set found on development set:")
    print(cv_model.best_params_)

def grid_search_forest(features, labels):
    """
    Grid search on random forest to find the best hyperparams.
    This is just an analysis tool.
    With this hyperparam, add it to the models.
    
    Note: It is kind of cheating that we optimize the hyperparams based on
    the same set we will later do kfold-evaluation on huh.

    Jan 14, 2018
    Out: Best parameters set found on development set:
        {'max_features': 2, 'n_estimators': 50}
    
    @param features, labels. X,Y of training set.
    """
    forest_params = {'n_estimators': [25, 50, 100, 250, 500],
                     'max_features': [2,3,5]}
    cv_model = GridSearchCV(\
                RandomForestClassifier( \
                        random_state=RANDOM_STATE), forest_params, cv=5,\
                       scoring='accuracy')
    cv_model.fit(features, labels)
    print("Best parameters set found on development set:")
    print(cv_model.best_params_)
    
def write_submission(model, train_features, train_labels, test_features,
                     passenger_ids): 
    """
    Write the final submission.
    @param model (Model) - the best learning model to use
    @param train_features (DataFrame) - the features to train model with
    @param train_labels (DataFrame) - the labels to train model with
    @param test_features (DataFrame) - the test features to predict against
    @param passenger_ids (List<Int>) - The passenger ids
    """
    model.fit(train_features, train_labels)
    final_labels = model.predict(test_features)
    submission = pd.DataFrame({
            "PassengerId": passenger_ids,
            "Survived": final_labels
            })
    submission.to_csv('output/submission.csv', index=False)

"""
Main
"""
raw_train_df = pd.read_csv(TRAIN_PATH)
raw_test_df = pd.read_csv(TEST_PATH)
train_df = raw_train_df.copy()
test_df = raw_test_df.copy()

train_features = onehot_categories(
        update_features(raw_train_df, train_df))
test_features = onehot_categories(
        update_features(raw_train_df, test_df))
train_labels = raw_train_df["Survived"]


# Enable if you want to see feature selection
"""
features = get_selective_features(train_features, train_labels)
print(features)
sys.exit()
"""

# Enable if you want to cross-check the features.
"""
plot_variable_importance(train_features, train_labels)
plot_pearson(pd.concat([train_features, train_labels]))
sys.exit()
"""

# Generate models
models = gen_models()

# Enable if you want to tune hyperparams on first layer
"""
grid_search_xgboost(train_features, train_labels)
sys.exit()
"""

# Enable if you want to see performance of first layer
"""
evaluate_models(models, KFOLD, train_features, train_labels)
sys.exit()
"""

train_feats_ensemble, test_feats_ensemble = \
    get_ensemble_feats(models, train_features, train_labels, test_features, \
                       KFOLD)
# Enable if you want to tune hyperparams on second layer
"""
grid_search_xgboost(train_feats_ensemble, train_labels)
sys.exit()
"""

final_model = gen_second_layer_model()

# Enable if you want to see performance of second layer
"""
final_models = {"final": final_model}
evaluate_models(final_models, KFOLD, train_feats_ensemble, train_labels)
sys.exit()
"""
# Final training and prediction
write_submission(final_model, train_feats_ensemble, train_labels,
                 test_feats_ensemble, raw_test_df["PassengerId"])