import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from matplotlib import pyplot as plt

# Set plotly renderer
rndr_type = "jupyterlab+png"
pio.renderers.default = rndr_type


class PCA(object):
    def __init__(self):
        self.U = None
        self.S = None
        self.V = None
    
    def center_data(self, data: np.ndarray) -> np.ndarray:
        average = np.mean(data, axis=0, keepdims=True)
        data_centered = data - average
        return data_centered

    def fit(self, X: np.ndarray) -> None:  # 5 points
        """
        Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
        You may use the numpy.linalg.svd function
        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V transpose

        Hint: np.linalg.svd by default returns the transpose of V
              Make sure you remember to first center your data by subtracting the mean of each feature.

        Args:
            X: (N,D) numpy array corresponding to a dataset

        Return:
            None

        Set:
            self.U: (N, min(N,D)) numpy array
            self.S: (min(N,D), ) numpy array
            self.V: (min(N,D), D) numpy array
        """
        X_centered = self.center_data(X)
        self.U, self.S, self.V = np.linalg.svd(X_centered, full_matrices=False)


    def transform(self, data: np.ndarray, K: int = 2) -> np.ndarray:  # 2 pts
        """
        Transform data to reduce the number of features such that final data (X_new) has K features (columns)
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            K: int value for number of columns to be kept

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data

        Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
        """
        X_centered = self.center_data(data)
        X_new = np.dot(X_centered, self.V[:K].T)
        return X_new
    
    def transform_rv(
        self, data: np.ndarray, retained_variance: float = 0.99
    ) -> np.ndarray:  # 3 pts
        """
        Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
        in X_new with K features
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            retained_variance: float value for amount of variance to be retained

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
                   to be kept to ensure retained variance value is retained_variance

        Hint: Make sure you remember to first center your data by subtracting the mean of each feature.

        """
        X_centered = self.center_data(data)
        cumulative_variance = np.cumsum(self.S ** 2) / np.sum(self.S ** 2)
        K = np.argmax(cumulative_variance >= retained_variance) + 1
        X_new = np.dot(X_centered, self.V[:K].T)
        return X_new
    
    def get_V(self) -> np.ndarray:
        """Getter function for value of V"""

        return self.V

    def visualize(self, X: np.ndarray, y: np.ndarray, fig_title) -> None:  # 5 pts
        """
        You have to plot two different scatterplots (2d and 3d) for this function. For plotting the 2d scatterplot, use your PCA implementation to reduce the dataset to only 2 features. You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
        Create a scatter plot of the reduced data set and differentiate points that have different true labels using color using plotly.
        Hint: Refer to https://plotly.com/python/line-and-scatter/ for making scatter plots with plotly.
        Hint: We recommend converting the data into a pandas dataframe before plotting it. Refer to https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html for more details.

        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,) numpy array, the true labels

        Return: None
        """
        self.fit(X)

        # Reduce dimensionailty to 2 using transform
        X_new = self.transform(data=X, K=2)

        df_2d = pd.DataFrame(data=X_new, columns=["x", "y"])
        df_2d["label"] = y
        plot_2d = px.scatter(
            df_2d,
            x="x",
            y="y",
            color="label",
            title=fig_title + " (2D)",
            labels={"x": "First PC", "y": "Second PC"},
        )

        # Reduce dimensionailty to 3 using transform
        X_new = self.transform(data=X, K=3)

        df_3d = pd.DataFrame(data=X_new, columns=["x", "y", "z"])
        df_3d["label"] = y

        plot_3d = px.scatter_3d(
            df_3d,
            x="x",
            y="y",
            z="z",
            color="label",
            title=fig_title + " (3D)",
            labels={
                "x" : "First PC",
                "y" : "Second PC",
                "z" : "Third PC",
            },
        )

        plot_2d.show()
        plot_3d.show()

