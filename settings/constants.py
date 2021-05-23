import os

DATA_FOLDER = 'data'
TRAIN_CSV = os.path.join(DATA_FOLDER, 'train.csv')
TEST_CSV = os.path.join(DATA_FOLDER, 'test.csv')
SAVED_ESTIMATOR = os.path.join('models', 'GridSearch.pickle')