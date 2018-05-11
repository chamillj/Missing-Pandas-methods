from sklearn import preprocessing
import pandas as pd
from collections import defaultdict


class DfOneHotEncoder():
    def __init__(self):
        self.le = defaultdict()
        self.ohe = defaultdict()
        self.dropped_columns = {}

    def fit(self, df, columns=None, **kwargs):

        self.le = df[columns].apply(lambda x: preprocessing.LabelEncoder().fit(x)).to_dict()

        for column in columns:
            self.ohe[column] = preprocessing.\
                OneHotEncoder(sparse=False).\
                fit(self.le[column].transform(df[column]).reshape(len(df[column]), 1))

    def transform(self, df, columns=None, drop_first=False, **kwargs):

        for column in columns:
            label_encode = self.le[column].transform(df[column]).reshape(len(df[column]), 1)
            one_hot_encode = self.ohe[column].transform(label_encode)

            df = df.drop(column, axis=1)
            new_col_name = [str(column) + "_" + str(cat_name) for cat_name in list(self.le[column].classes_)]

            if drop_first:
                start_index = 1
                self.dropped_columns[column] = new_col_name[0]

            else:
                start_index = 0

            df = pd.concat([df, pd.DataFrame(one_hot_encode[:,start_index:], columns=new_col_name[start_index:])], axis=1)

        return df

    def fit_transform(self, df, columns=None, **kwargs):
        self.fit(df, columns, **kwargs)
        return self.transform(df, columns, **kwargs)

    def inverse_transform(self, df, columns):
        for column in columns:
            names_of_encoded_columns = [i for i in list(df) if i.startswith(column + "_")]

            df[column] = df[names_of_encoded_columns].\
                apply(lambda x: x[x != 0].keys()[0][len(column)+1:] if ~(x == 0).all()
                    else self.dropped_columns[column][len(column)+1:], axis=1)

            df = df.drop(names_of_encoded_columns,axis=1)

        return df
