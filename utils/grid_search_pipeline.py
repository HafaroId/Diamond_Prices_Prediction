from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


class GridSearchPipeline:
    """
        A class used to construct GridSearch
        ...
        Attributes
        __________
        X_train : pandas DataFrame
            train dataset
        y_train : pandas DataFrame
            target dataset
        Methods
        -------
        rfc_grid_search()
            construct Pipelines based on RandomForestClassifier
        """

    def grid_search_pipeline(self, X_train, y_train, model=RandomForestRegressor(), param_grid=None, cv=5):
        """
        Construct Pipelines for numerical and categorical columns.
        Create self.preprocessor and self.preprocessor_dict for final Pipeline and further GridSearch
        """

        # At first we'll find columns with numerical and categorical values
        num_features = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
        cat_features = [col for col in X_train.columns if X_train[col].dtype == 'object']

        # Pipeline for numerical features that contains scaling and normalization operations
        num_transformer = Pipeline(steps=[
            ('scaler', 'passthrough'),
            ('norm', 'passthrough')
        ])

        # Pipeline for categorical features that contains encoding operation
        cat_transformer = Pipeline(steps=[
            ('encode', 'passthrough')
        ])

        # Combining of num and cat Pipelines into one preprocessor step using ColumnTransformer.
        # This preprocessor will be used in final Pipeline and further in GridSearch
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])

        # Constructing of final Pipeline that contains preprocessor and regression model
        final_estimator = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        # Final GridSearch construction. As estimator parameter of GridSearch we'll pass our final Pipeline:
        grid_search = GridSearchCV(final_estimator, param_grid=param_grid,
                                   scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        return -grid_search.best_score_, grid_search.best_estimator_, grid_search.best_params_
