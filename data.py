# written by David Sommer 2019 to illustrate ML to a friend

import pandas as pd
import numpy as np

def read_data(filename):
    df = pd.read_csv(filename, sep=';')
    df = df.mask(df.astype(object).eq('None')).dropna()

    X = df.loc[:, ['dayofyear', 'minuteofday', 't_bern_0', 't_thun_0']]
    y = df[('t_bern_90')]

    return X, y


def import_data_classes(filename, n_classes):
    X, y = read_data(filename)

    delta = X.loc[:,'t_bern_0'] - y

    da_min = np.min(delta)
    da_max = np.max(delta)

    bins = np.linspace(da_min, da_max, n_classes)
    scalefactor = bins[1] - bins[0]

    print("bins", bins)

    X = X.to_numpy()
    binned_y = np.digitize(delta, bins)


    return X, binned_y, scalefactor

def import_data_contineous(filename):
    X, y = read_data(filename)

    X = X.to_numpy()
    y = y.to_numpy()

    return X, y


if __name__ == "__main__":
    # X,y = import_data_contineous(filename="data.csv/data_v01.csv")
    X,y, scale = import_data_classes(filename="data.csv/data_v01.csv", n_classes=20)
    print(X,y)