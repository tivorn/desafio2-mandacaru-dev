import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

def prepare_data(file_path, return_split=True):
    df = pd.read_csv(file_path)
    df = df[['Email', 'Label']]
    #df = df.dropna(axis=0)

    docs, labels = df.Email.values, df.Label.values

    labels = encoder.fit_transform(labels)

    if return_split:
        return train_test_split(
            docs,
            labels,
            test_size=0.25,
            random_state=1
        )

    return docs, labels

def get_class_names(y):
    return np.unique(encoder.inverse_transform(y))