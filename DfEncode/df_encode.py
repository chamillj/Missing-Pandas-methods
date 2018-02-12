
from sklearn import preprocessing
import pandas as pd
from collections import defaultdict


class DfEncoder():
    """Encode categorical columns either using one hot encoding or dummy encoding.
    The input is a Padas data frame and list of columns that we would like to encode.
    The same can be done with Pandas get_dummy method. However, the get_dummy method does not
    provide the ability to use the same encoding scheme for two datasets, e.g across train
    and datasets, or the possibility of inverse transform

    Can use either one-hot encoding ie 1 to k mapping or dummy encoding i.e 1 to k-1

    Example
    _________

    >>raw_data.head()
    Survived	Pclass	Sex	Age	SibSp	Parch	Fare	Embarked	Famsize
    0	0	3	male	22.0	1	0	7.2500	S	2
    1	1	1	female	38.0	1	0	71.2833	C	2
    2	1	3	female	26.0	0	0	7.9250	S	1
    3	1	1	female	35.0	1	0	53.1000	S	2
    4	0	3	male	35.0	0	0	8.0500	S	1

    >>DFE = DfOneHotEncoder()
    >>test= DFE.fit_transform(raw_data, ["Embarked", "Sex"], drop_first=True)
    >>test.head()

    Survived	Pclass	Age	SibSp	Parch	Fare	Famsize	Embarked_Q	Embarked_S	Sex_male
    0	0	3	22.0	1	0	7.2500	2	0.0	1.0	1.0
    1	1	1	38.0	1	0	71.2833	2	0.0	0.0	0.0
    2	1	3	26.0	0	0	7.9250	1	0.0	1.0	0.0
    3	1	1	35.0	1	0	53.1000	2	0.0	1.0	0.0
    4	0	3	35.0	0	0	8.0500	1	0.0	1.0	1.0

    >>test_inverse = DFE.inverse_transform(test, ["Embarked", "Sex"])
    >>test_inverse.head()
    Survived	Pclass	Age	SibSp	Parch	Fare	Famsize	Embarked	Sex
    0	0	3	22.0	1	0	7.2500	2	S	male
    1	1	1	38.0	1	0	71.2833	2	C	female
    2	1	3	26.0	0	0	7.9250	1	S	female
    3	1	1	35.0	1	0	53.1000	2	S	female
    4	0	3	35.0	0	0	8.0500	1	S	male

     """

    def __init__(self):
        self.le = defaultdict()  # store label encoder classes
        self.ohe = defaultdict() # store one hot encoder classes
        self.dropped_columns = {}  # if the first column is dropped, store its name

    def fit(self, df, columns=None, **kwargs):

        """Fit DfEncode to X.
        Parameters
        ----------
        X : Pandas DataFrame
        columns: list of columns to be fitted

        Returns
        -------
        self

        """

        if not columns:
            columns = list(df.columns)

        self.le = df[columns].apply(lambda x: preprocessing.LabelEncoder().fit(x)).to_dict()

        for column in columns:
            self.ohe[column] = preprocessing.\
                OneHotEncoder(sparse=False).\
                fit(self.le[column].transform(df[column]).reshape(len(df[column]), 1))

    def transform(self, df, columns=None, drop_first=False, **kwargs):

        """Encode X.

        Parameters
        ----------
        X : Pandas DataFrame
        columns: list of columns to be encoded
        drop_first : True/False True for 1 to k-1 encode

        Returns
        -------
        Pandas dataframe with new columns added for encoded variable. Column names are column_name + value

        """

        for column in columns:
            label_encode = self.le[column].transform(df[column]).reshape(len(df[column]), 1)
            one_hot_encode = self.ohe[column].transform(label_encode)

            df = df.drop(column, axis=1)
            new_col_name = [column + "_" + cat_name for cat_name in list(self.le[column].classes_)]

            if drop_first:
                start_index = 1
                self.dropped_columns[column] = new_col_name[0]

            else:
                start_index = 0

            df = pd.concat([df, pd.DataFrame(one_hot_encode[:,start_index:], columns=new_col_name[start_index:])], axis=1)

        return df

    def fit_transform(self, df, columns=None, **kwargs):
        """
        Short cut for doing fit and transform in one method call
        Parameters
        ----------
        X : Pandas DataFrame
        columns: list of columns to be encoded
        drop_first : True/False True for 1 to k-1 encode

        Returns
        -------
        Pandas dataframe with new columns added for encoded variable. Column names are column_name + value
        """

        self.fit(df, columns, **kwargs)
        return self.transform(df, columns, **kwargs)

    def inverse_transform(self, df, columns):
        """Encode X.
        Parameters
        ----------
        X : Pandas DataFrame
        columns: list of columns to be encoded

        Returns
        -------
        Pandas dataframe that is the inverse transform of the orginal

        """

        for column in columns:
            names_of_encoded_columns = [i for i in list(df) if i.startswith(column + "_")]

            df[column] = df[names_of_encoded_columns].\
                apply(lambda x: x[x != 0].keys()[0][len(column)+1:] if ~(x == 0).all()
                    else self.dropped_columns[column][len(column)+1:], axis=1)

            df = df.drop(names_of_encoded_columns,axis=1)

        return df
