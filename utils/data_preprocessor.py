class DataLoader(object):
    """
    A class used to preprocess data before modeling ir predicting operations
    ...
    Attributes
    __________
    dataset : pandas DataFrame
    Methods
    -------
    preprocess()
        split dataset to X and y
        preprocess data before modeling or predicting
    """
    def __init__(self, dataset):
        self.dataset = dataset.copy()

    def preprocess(self):
        # drop outliers
        self.dataset.drop(self.dataset[self.dataset["x"] == 0].index, inplace=True)
        self.dataset.drop(self.dataset[self.dataset["y"] == 0].index, inplace=True)
        self.dataset.drop(self.dataset[self.dataset["z"] == 0].index, inplace=True)
        self.dataset.drop(self.dataset[self.dataset["y"] > 30].index, inplace=True)
        self.dataset.drop(self.dataset[self.dataset["x"] > 30].index, inplace=True)

        X = self.dataset.drop(['depth', 'table', 'price'], axis=1)  # X assignment
        y = self.dataset['price']  # target assignment
        return X, y
