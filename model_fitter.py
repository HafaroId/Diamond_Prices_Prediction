import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, \
    MinMaxScaler, PowerTransformer, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

from utils.data_preprocessor import DataLoader
from settings.constants import TRAIN_CSV
from utils.grid_search_pipeline import GridSearchPipeline

# read data
train = pd.read_csv(TRAIN_CSV)

# preprocess train data before fit operation
dataloader = DataLoader(train)
X_train, y_train = dataloader.preprocess()

print(X_train.head())

# call GridSearchEstimator class in order to find best estimator
estimator = GridSearchPipeline()

# define GridSearch best_parameters
cv = 4  # number of cross validation runs
model = RandomForestRegressor()  # RandomForest model initialization

# categories for OrdinalEncoder
cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_categories = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

# dict with preprocessing parameters
preprocessor_dict = dict(preprocessor__num__scaler=[StandardScaler(), MinMaxScaler()],
                         preprocessor__num__norm=[PowerTransformer()],
                         preprocessor__cat__encode=[OrdinalEncoder(categories=[cut_categories,
                                                                               color_categories,
                                                                               clarity_categories]),
                                                    OneHotEncoder()]
                         )

# dict with RandomForest model hyperparameters
RFC_dict = dict(regressor__n_estimators=[100, 300],
                regressor__min_samples_leaf=[2, 4])
RFC_dict.update(preprocessor_dict)  # merge preprocessing and RF dicts into one dictionary

# find out best parameters by GridSearch
best_score, best_estimator, best_params = estimator.grid_search_pipeline(X_train, y_train, param_grid=RFC_dict,
                                                                         model=model, cv=cv)

# dump best estimator to GridSearch.pickle file
with open('models/GridSearch.pickle', 'wb')as f:
    pickle.dump(best_estimator, f)

print("Best GridSearch MAE:", best_score)
print("Best estimator params:", best_params)
