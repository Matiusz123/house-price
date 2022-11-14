import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 90)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import scipy.stats
from sklearn.preprocessing import StandardScaler
#from pycaret.regression import setup, compare_models
from sklearn.model_selection import KFold, cross_val_score
import optuna

path_train = "/Users/mateusz/Downloads/train(1).csv"
path_test = "/Users/mateusz/Downloads/test(1).csv"

train_x = pd.read_csv(path_train)
test_x = pd.read_csv(path_test)

target = train_x['SalePrice']
test_ids = test_x['Id']

train0 = train_x.drop(['Id', 'SalePrice'], axis=1)
test0 = test_x.drop('Id', axis=1)

alldata = pd.concat([train0, test0], axis=0).reset_index(drop=True)

# Reseting MSSubClass to right format

alldata['MSSubClass'] = alldata['MSSubClass'].astype(str)

# Filling the Categorical Missing Values

for column in [
    'Alley',
    'BsmtQual',
    'BsmtCond',
    'BsmtExposure',
    'BsmtFinType1',
    'BsmtFinType2',
    'FireplaceQu',
    'GarageType',
    'GarageFinish',
    'GarageQual',
    'GarageCond',
    'PoolQC',
    'Fence',
    'MiscFeature'
]:
    alldata[column] = alldata[column].fillna('None')

for column in [
    'MSZoning',
    'Utilities',
    'Exterior1st',
    'Exterior2nd',
    'MasVnrType',
    'Electrical',
    'KitchenQual',
    'Functional',
    'SaleType'
]:
    alldata[column] = alldata[column].fillna(alldata[column].mode()[0])

alldata1 = alldata.copy()

# Numerical Value prep

num_data = alldata1.select_dtypes(np.number)
non_na_column = num_data.loc[:, num_data.isna().sum() > 0]

for column in non_na_column.columns:
    alldata1[column] = alldata1[column].fillna(alldata1[str(column)].describe()['mean'])

#Feature Engineering
alldata1["SqFtPerRoom"] = alldata1["GrLivArea"] / (alldata1["TotRmsAbvGrd"] +
                                                       alldata1["FullBath"] +
                                                       alldata1["HalfBath"] +
                                                       alldata1["KitchenAbvGr"])

alldata1['Total_Home_Quality'] = alldata1['OverallQual'] + alldata1['OverallCond']

alldata1['Total_Bathrooms'] = (alldata1['FullBath'] + (0.5 * alldata1['HalfBath']) +
                               alldata1['BsmtFullBath'] + (0.5 * alldata1['BsmtHalfBath']))

alldata1["HighQualSF"] = alldata1["1stFlrSF"] + alldata1["2ndFlrSF"]

# Skew fix

alldata2 = alldata1.copy()
skew_data = pd.DataFrame(num_data.columns, columns= ['Feature'])
skew_data['Skew'] = skew_data['Feature'].apply(lambda feature: scipy.stats.skew(alldata2[feature]))
skew_data['Abs Skew'] = skew_data['Skew'].apply(abs)
skew_data['Skewed'] = skew_data['Abs Skew'].apply(lambda x: True if x >= 0.5 else False)

for column in skew_data.query('Skewed == True')['Feature'].values:
    alldata2[column] = np.log1p(alldata2[column])

# Month fix

alldata2['MoSold'] = (-np.cos(0.5236) * alldata2['MoSold'])

alldata3 = alldata2.copy()

# Encode Categorical Values

alldata3 = pd.get_dummies(alldata3)

# Scaling

scaler = StandardScaler()
scaler.fit(alldata3)

alldata3 = pd.DataFrame(scaler.transform(alldata3), index=alldata3.index, columns=alldata3.columns)

# Print target and fix the skew

# plt.figure(figsize=(20, 10))
#
# plt.subplot(1, 2, 1)
# sns.distplot(target, kde=True, fit=scipy.stats.norm)
# plt.title("Without Log Transform")
#
# plt.subplot(1, 2, 2)
# sns.distplot(np.log(target), kde=True, fit=scipy.stats.norm)
# plt.xlabel("Log SalePrice")
# plt.title("With Log Transform")
#
# plt.show()

log_target = np.log(target)

train_final = alldata3.loc[:train0.index.max(), :].copy()
test_final = alldata3.loc[train0.index.max() + 1:, :].reset_index(drop=True).copy()

# Model selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, Ridge, OrthogonalMatchingPursuit

# It has to be done in colab because pycharm does not have cababilities
# used function
# # _ = setup(data=pd.concat([train_final, log_target], axis=1), target='SalePrice')
# compare_models()

# Hyperparameter Optimization
kf = KFold(n_splits=10)
# def br_objective(trial):
#     n_iter = trial.suggest_int('n_iter', 50, 600)
#     tol = trial.suggest_float('tol', 1e-8, 10.0)
#     alpha_1 = trial.suggest_float('alpha_1', 1e-8, 10.0)
#     alpha_2 = trial.suggest_float('alpha_2', 1e-8, 10.0)
#     lambda_1 = trial.suggest_float('lambda_1', 1e-8, 10.0)
#     lambda_2 = trial.suggest_float('lambda_2', 1e-8, 10.0)
#
#     model = BayesianRidge(
#         n_iter=n_iter,
#         tol=tol,
#         alpha_1=alpha_1,
#         alpha_2=alpha_2,
#         lambda_1=lambda_1,
#         lambda_2=lambda_2
#     )
#     model.fit(train_final, log_target)
#
#     cv_scores = np.exp(np.sqrt(-cross_val_score(model, train_final, log_target, scoring='neg_mean_squared_error', cv=kf)))
#
#     return np.mean(cv_scores)
#
#
# study = optuna.create_study(direction='minimize')
# study.optimize(br_objective, n_trials=100)
# print(study.best_params)

# def gradient_objective(trial):
#     learning_rate = trial.suggest_float('learning_rate', 0.1, 100.0)
#     n_estimators = trial.suggest_int('n_estimators', 1, 100)
#     subsample = trial.suggest_float('subsample', 0.0, 1.0)
#     criterion = trial.suggest_categorical('criterion', ['friedman_mse', 'mse'])
#     min_samples_split = trial.suggest_int('min_samples_split', 2, 100)
#     min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 100)
#     min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5)
#     max_depth = trial.suggest_int('max_depth', 1, 100)
#     min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0.1, 100.0)
#     alpha = trial.suggest_float('alpha', 0.0, 1.0)
#     tol = trial.suggest_float('tol', 0.1, 100.0)
#
#     model = GradientBoostingRegressor(
#         learning_rate=learning_rate,
#         n_estimators=n_estimators,
#         subsample=subsample,
#         criterion=criterion,
#         min_samples_split=min_samples_split,
#         min_samples_leaf=min_samples_leaf,
#         min_weight_fraction_leaf=min_weight_fraction_leaf,
#         max_depth=max_depth,
#         min_impurity_decrease=min_impurity_decrease,
#         alpha=alpha,
#         tol=tol
#     )
#     model.fit(train_final, log_target)
#
#     cv_scores = np.exp(np.sqrt(-cross_val_score(model, train_final, log_target, scoring='neg_mean_squared_error', cv=kf)))
#
#     return np.mean(cv_scores)
#
#
# study = optuna.create_study(direction='minimize')
# study.optimize(gradient_objective, n_trials=100)
# print(study.best_params)

# def ridge_objective(trial):
#     alpha = trial.suggest_float('alpha', 0.0, 1000.0)
#
#     model = Ridge(
#         alpha=alpha,
#     )
#     model.fit(train_final, log_target)
#
#     cv_scores = np.exp(np.sqrt(-cross_val_score(model, train_final, log_target, scoring='neg_mean_squared_error', cv=kf)))
#
#     return np.mean(cv_scores)
#
#
# study = optuna.create_study(direction='minimize')
# study.optimize(ridge_objective, n_trials=100)
# print(study.best_params)

br_params = {'n_iter': 527,
             'tol': 9.32896955782112,
             'alpha_1': 8.605523008676759,
             'alpha_2': 9.562469665347267,
             'lambda_1': 2.1820733815100217,
             'lambda_2': 0.0013799137961379299}
gradient_params = {'learning_rate': 0.2951715584710891,
                   'n_estimators': 23,
                   'subsample': 0.3519380187083131,
                   'criterion': 'friedman_mse',
                   'min_samples_split': 90,
                   'min_samples_leaf': 59,
                   'min_weight_fraction_leaf': 0.20440206063820807,
                   'max_depth': 6,
                   'min_impurity_decrease': 10.358064513448895,
                   'alpha': 0.9503314966107902,
                   'tol': 79.8041067368768}
ridge_params = {'alpha': 350.4538383367691}

models = {
    'gradient': GradientBoostingRegressor(**gradient_params),
    "br": BayesianRidge(**br_params),
    "ridge": Ridge(**ridge_params),
    "omp": OrthogonalMatchingPursuit()
}

for name, model in models.items():
    model.fit(train_final, log_target)
    print(name + " trained.")

results = {}

for name, model in models.items():
    result = np.exp(np.sqrt(-cross_val_score(model, train_final, log_target, scoring='neg_mean_squared_error', cv=kf)))
    results[name] = result

for name, result in results.items():
    print("----------\n" + name)
    print(np.mean(result))
    print(np.std(result))

final_predictions = (
    0.25 * np.exp(models['gradient'].predict(test_final)) +
    0.25 * np.exp(models['br'].predict(test_final)) +
    0.25 * np.exp(models['ridge'].predict(test_final)) +
    0.25 * np.exp(models['omp'].predict(test_final))
)


submission = pd.concat([test_ids, pd.Series(final_predictions, name='SalePrice')], axis=1)

submission.to_csv("submission1.csv", index=False)


# score

# 9.45950 - one model, no hyper parameters
# 0.12822 - four models, no hyper parameters
# 0.14407 - four models, with hyper parameters
# 0.16642 - four models, with weights and hyper parameters
# 0.16310 - feature engineering
