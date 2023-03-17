#!/usr/bin/env python3

# tools from sklearn
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

lb = LabelBinarizer()

def load_data():
    # load and separate data and labels
    data, labels = fetch_openml('mnist_784', version=1, return_X_y=True)
    # normalise data
    data = data.astype("float")/255.0
    # split data
    (X_train, X_test, y_train, y_test) = train_test_split(data,
                                                        labels, 
                                                        test_size=0.2)
    # convert labels to one-hot encoding
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    return X_train, X_test, y_train, y_test

def train_model(train_data, label_data):
    model = Sequential() # defining a sequential model, meaning a feed-forward neural network
    model.add(Dense(256, # adding a dense (fully-connected) layer with a first layer of 256 nodes
                    input_shape=(784,), # size of input layer
                    activation="relu")) # activated by ReLU, not logarithmic
    model.add(Dense(128, # another hidden layer, 128 nodes
                    activation="relu"))
    model.add(Dense(10,  # output layer, 10 nodes
                    activation="softmax")) # activation function predicting 1 or 0 for final output
    model.summary()
    # train model using SGD, stochastic gradient descent
    sgd = SGD(0.01) # the higher the value, the quicker the model's attempts to learn/optimize
    model.compile(loss="categorical_crossentropy", # compile makes it a computational graph structure, loss= defines which loss function is used
                optimizer=sgd, # use the previously defined sgd
                metrics=["accuracy"]) # what the model aims to optimize, like recall, precision, f1-score, accuracy etc.
    model.fit(train_data, label_data, # creating history, saving the history of our model
                epochs=10, 
                batch_size=32) # update model weights after learning from a whole batch (of 32) rather than after each picture
    return model

def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test, batch_size=32)
    print(classification_report(y_test.argmax(axis=1), # for y-test, give the index of the largest argument
                            predictions.argmax(axis=1), # same for predictions
                            target_names=[str(x) for x in lb.classes_]))
    return predictions

if __name__ == "__main__":
    main()