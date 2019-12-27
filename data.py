import pandas as pd


def import_data(filename, n_classes):
    #   pd.read_csv(filename, parse_dates=True)

    headers = ['date', 'temp_bern', 'temp_thun']
    dtypes = {'date': 'str', 'col2': 'float', 'col3': 'float'}
    parse_dates = ['date']
    ret = pd.read_csv(filename, sep=';', header=None, names=headers, dtype=dtypes, parse_dates=parse_dates)
    ret = ret.mask(ret.astype(object).eq('None')).dropna()

    print(ret)

    return X, y



if __name__ == "__main__":
    X,y = import_data(filename="hydro_test.csv", n_classes=20)
    print(X,y)