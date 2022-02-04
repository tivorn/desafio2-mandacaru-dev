import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding, Conv1D, Flatten, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import LSTM

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer, Continuous

classifiers = {
    "SVC": SVC(probability=True),
    "LogisticRegression": LogisticRegression(max_iter=10000)
}

params_grid = {
        "SVC": {
            "C": Continuous(0,10),
            "gamma": Continuous(0,10),
            "kernel": Categorical(["linear", "rbf"])
        },
        "LogisticRegression": {
            "C": Continuous(0,10),
            "penalty": Categorical(["l2"])
        }
}

def evolved_classifier(
    X_train, X_val, y_val, y_train,
    classifier_name, best_params=False
):
    classifier = classifiers[classifier_name]
    cv = KFold(n_splits=10, shuffle=True)

    params = params_grid[classifier_name] 

    evolved_estimator = GASearchCV(
        estimator=classifier,
        cv=cv,
        scoring="accuracy",
        population_size=15,
        generations=30,
        crossover_probability=0.5,
        mutation_probability=0.1,
        param_grid=params,
        algorithm="eaSimple",
        verbose=True
    )

    evolved_estimator.fit(X_train, y_train)

    return evolved_estimator

def build_DNN(
    vocab_size,
    maxlen,
    embedding_matrix,
    hidden_dims,
    dropout_rate=0.2
):
    model=Sequential()
    model.add(Embedding(
        vocab_size,
        100,
        input_length=maxlen,
        weights=[embedding_matrix],
        trainable=False
    ))
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))

    return model


def build_LSTM(
    vocab_size,
    maxlen,
    embedding_matrix,
    hidden_dims
):
    model = Sequential()
    model.add(Embedding(
        vocab_size,
        100,
        input_length=maxlen,
        weights=[embedding_matrix],
        trainable=False
    ))
    model.add(LSTM(hidden_dims, activation="relu"))
    model.add(Dense(3, activation="softmax"))
    
    return model

def build_Conv1D(
    vocab_size,
    maxlen,
    embedding_matrix,
    filters,
    kernel_size,
    hidden_dims
):
    model = Sequential()
    model.add(Embedding(
        vocab_size,
        100,
        input_length=maxlen,
        weights=[embedding_matrix],
        trainable=False
    ))
    model.add(Dropout(0.5))
    model.add(Conv1D(
        filters,
        kernel_size,
        padding='valid',
        activation='relu'
    ))
    model.add(MaxPooling1D())
    model.add(Conv1D(
        filters,
        kernel_size,
        padding='valid',
        activation='relu'
    ))
    model.add(MaxPooling1D())
    model.add(Conv1D(
        filters,
        kernel_size,
        padding='valid',
        activation='relu'
    ))
    model.add(GlobalMaxPooling1D())
    #model.add(Flatten())
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    return model


