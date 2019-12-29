# written by David Sommer 2019 to illustrate ML to a friend

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from data import import_data_contineous, import_data_classes


def accuracy_metric_classes(y_hat, y, scalefactor=1):
    arr = np.abs(y_hat - y, dtype=np.float64)

    da_mean = np.mean(arr)
    da_stdDev = np.sqrt(np.var(arr))

    return scalefactor*da_mean, scalefactor*da_stdDev


def accuracy_metric_contineous(y_hat, y):
    err = np.abs(y_hat - y)

    da_mean = np.mean(err)
    da_stdDev = np.sqrt(np.var(err))

    return da_mean, da_stdDev


def example_apply_linear_regression(filename):
    X ,y = import_data_contineous(filename)

    X = X[:,(2,3)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    reg = LinearRegression().fit(X_train, y_train)

    y_hat_train = reg.predict(X_train)
    y_hat_test = reg.predict(X_test)

    acc_train = accuracy_metric_contineous(y_hat_train, y_train)
    acc_test = accuracy_metric_contineous(y_hat_test, y_test)

    print(acc_train)
    print(acc_test)
    print(np.mean(np.abs(X_train[:,0] - y_train)))


def example_apply_other_standard_algorithms(filename):
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]


    NUM_CLASSES=50

    DATA_CUT = 1000
    X, y, scalefactor = import_data_classes(filename, NUM_CLASSES)
    print("scalefactor", scalefactor)
    X, y = X[:DATA_CUT], y[:DATA_CUT]

    if False: # if you want to plot the data
        plt.plot(X[:,2],X[:,3], ".", label="2,3")
        plt.plot(X[:,2],y, ".", label="2,y")
        plt.plot(X[:,3],y, ".", label="3,y")
        plt.plot(X[:,2] - X[:,3],y, ".", label="2-3,y")
        plt.legend()
        plt.show()

    datasets = ((X,y),)

    for i, data in enumerate(datasets):
        print("i", i)
        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)  # random_state=42
        for j, classifier in enumerate(classifiers):
            print("y", j)
            classifier.fit(X_train, y_train)

            y_hat_train = classifier.predict(X_train)
            y_hat_test = classifier.predict(X_test)



            acc_mean_train, acc_stdDev_train = accuracy_metric_classes(y_hat_train, y_train, scalefactor)
            print(names[j], acc_mean_train, acc_stdDev_train)
            acc_mean_test, acc_stdDev_test = accuracy_metric_classes(y_hat_test, y_test, scalefactor)
            print(names[j], acc_mean_test, acc_stdDev_test)


def example_neural_network(filename):
    from keras.models import Sequential
    from keras.layers import Dense

    # our (classifiable) data
    X, y, scalefactor = import_data_classes(filename, n_classes=50)
    print("scalefactor", scalefactor)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    # define the keras model
    model = Sequential([
        Dense(10, input_dim=len(X[0]), activation='relu'),
        Dense(20, activation='relu'),
        Dense(60, activation='relu'),
        Dense(30, activation='relu'),
        Dense(30, activation='relu'),
        Dense(30, activation='relu'),
        Dense(np.max(y)+1, activation='sigmoid')
        ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train,y_train, epochs=100, batch_size=50)

    y_hat_train = np.argmax(model.predict(X_train), axis=1)
    y_hat_test = np.argmax(model.predict(X_test), axis=1)


    acc_mean_train, acc_stdDev_train = accuracy_metric_classes(y_hat_train, y_train, scalefactor)
    print(acc_mean_train, acc_stdDev_train)
    acc_mean_test, acc_stdDev_test = accuracy_metric_classes(y_hat_test, y_test, scalefactor)
    print(acc_mean_test, acc_stdDev_test)


if __name__ == "__main__":
    FILENAME="data.csv/data_v01.csv"

    example_apply_linear_regression(FILENAME)
    example_apply_other_standard_algorithms(FILENAME)
    example_neural_network(FILENAME)