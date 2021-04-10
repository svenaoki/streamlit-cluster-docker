import streamlit as st
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


class Predictor():
    # get data
    def get_data(self):
        file = st.sidebar.file_uploader("Choose file (csv)", type="csv")
        if file is not None:
            self.data = pd.read_csv(file)
            return self.data
        else:
            st.success("Please upload a CSV File")

    # choose classifier
    def select_algo(self):
        self.type = st.sidebar.selectbox("Algorithm", ("Classification", "Clustering"))
        if self.type == "Classification":
            self.chosen_classifier = st.sidebar.selectbox(
                "Type of Classifier", ("Random Forest", "Logistic Regression")
            )
            if self.chosen_classifier == "Random Forest":
                self.n_trees = st.sidebar.slider("number of trees", 1, 1000, 1)
            elif self.chosen_classifier == "Logistic Regression":
                self.max_iter = st.sidebar.slider(
                    "Choose maximum number of Iterations", 1, 50, 10
                )
        if self.type == "Clustering":
            pass

    # select features and targets
    def set_features(self):
        self.features = st.sidebar.multiselect(
            "Please choose your features", self.data.columns
        )
        self.target = st.sidebar.selectbox(
            "Please choose your target variable",
            [
                variable
                for variable in self.data.columns
                if variable not in self.features
            ],
        )

    # preprocess data
    def preprocess_data(self, test_size):
        X = self.data[self.features]
        # X = MinMaxScaler().fit_transform(X)
        y = self.data[self.target]
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(
            X, y, test_size=test_size
        )

    # plot some data
    def plot_data(self):
        cols = st.multiselect("Choose variables to see", self.data.columns)
        if len(cols) > 1:
            fig, ax = plt.subplots()
            sns.scatterplot(
                x=cols[0],
                y=cols[1],
                hue=self.target,
                data=self.data,
                ax=ax,
            )
            st.pyplot(fig)

    # make prediction


if __name__ == "__main__":
    controller = Predictor()
    controller.data = controller.get_data()
    if controller.data is not None:
        controller.select_algo()
        controller.set_features()
    if controller.data is not None and len(controller.features) > 1:
        test_size = st.sidebar.slider("Choose size (%) of test set", 1, 99, 33)
        controller.preprocess_data(test_size)
        controller.plot_data()
