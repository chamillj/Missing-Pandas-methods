 # DFEncode

  Encode categorical columns either using one hot encoding or dummy encoding.
  The input is a Padas data frame and list of columns that we would like to encode.
  The same can be done with Pandas get_dummy method. However, the get_dummy method does not
  provide the ability to use the same encoding scheme for two datasets, e.g across train
  and datasets, or the possibility of inverse transform

  Can use either one-hot encoding ie 1 to k mapping or dummy encoding i.e 1 to k-1

  Example
  _________

  ```python
  >>raw_data.head()
  ```  
  <div>
  <style scoped>
      .dataframe tbody tr th:only-of-type {
          vertical-align: middle;
      }

      .dataframe tbody tr th {
          vertical-align: top;
      }

      .dataframe thead th {
          text-align: right;
      }
  </style>
  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: right;">
        <th></th>
        <th>Survived</th>
        <th>Pclass</th>
        <th>Sex</th>
        <th>Age</th>
        <th>SibSp</th>
        <th>Parch</th>
        <th>Fare</th>
        <th>Embarked</th>
        <th>Famsize</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>0</th>
        <td>0</td>
        <td>3</td>
        <td>male</td>
        <td>22.0</td>
        <td>1</td>
        <td>0</td>
        <td>7.2500</td>
        <td>S</td>
        <td>2</td>
      </tr>
      <tr>
        <th>1</th>
        <td>1</td>
        <td>1</td>
        <td>female</td>
        <td>38.0</td>
        <td>1</td>
        <td>0</td>
        <td>71.2833</td>
        <td>C</td>
        <td>2</td>
      </tr>
      <tr>
        <th>2</th>
        <td>1</td>
        <td>3</td>
        <td>female</td>
        <td>26.0</td>
        <td>0</td>
        <td>0</td>
        <td>7.9250</td>
        <td>S</td>
        <td>1</td>
      </tr>
      <tr>
        <th>3</th>
        <td>1</td>
        <td>1</td>
        <td>female</td>
        <td>35.0</td>
        <td>1</td>
        <td>0</td>
        <td>53.1000</td>
        <td>S</td>
        <td>2</td>
      </tr>
      <tr>
        <th>4</th>
        <td>0</td>
        <td>3</td>
        <td>male</td>
        <td>35.0</td>
        <td>0</td>
        <td>0</td>
        <td>8.0500</td>
        <td>S</td>
        <td>1</td>
      </tr>
    </tbody>
  </table>
  </div>

  ```python
  test= DFE.fit_transform(raw_data, ["Embarked", "Sex"], drop_first=True)
  ```


  ```python
  test.head()
  ```




  <div>
  <style scoped>
      .dataframe tbody tr th:only-of-type {
          vertical-align: middle;
      }

      .dataframe tbody tr th {
          vertical-align: top;
      }

      .dataframe thead th {
          text-align: right;
      }
  </style>
  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: right;">
        <th></th>
        <th>Survived</th>
        <th>Pclass</th>
        <th>Age</th>
        <th>SibSp</th>
        <th>Parch</th>
        <th>Fare</th>
        <th>Famsize</th>
        <th>Embarked_Q</th>
        <th>Embarked_S</th>
        <th>Sex_male</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>0</th>
        <td>0</td>
        <td>3</td>
        <td>22.0</td>
        <td>1</td>
        <td>0</td>
        <td>7.2500</td>
        <td>2</td>
        <td>0.0</td>
        <td>1.0</td>
        <td>1.0</td>
      </tr>
      <tr>
        <th>1</th>
        <td>1</td>
        <td>1</td>
        <td>38.0</td>
        <td>1</td>
        <td>0</td>
        <td>71.2833</td>
        <td>2</td>
        <td>0.0</td>
        <td>0.0</td>
        <td>0.0</td>
      </tr>
      <tr>
        <th>2</th>
        <td>1</td>
        <td>3</td>
        <td>26.0</td>
        <td>0</td>
        <td>0</td>
        <td>7.9250</td>
        <td>1</td>
        <td>0.0</td>
        <td>1.0</td>
        <td>0.0</td>
      </tr>
      <tr>
        <th>3</th>
        <td>1</td>
        <td>1</td>
        <td>35.0</td>
        <td>1</td>
        <td>0</td>
        <td>53.1000</td>
        <td>2</td>
        <td>0.0</td>
        <td>1.0</td>
        <td>0.0</td>
      </tr>
      <tr>
        <th>4</th>
        <td>0</td>
        <td>3</td>
        <td>35.0</td>
        <td>0</td>
        <td>0</td>
        <td>8.0500</td>
        <td>1</td>
        <td>0.0</td>
        <td>1.0</td>
        <td>1.0</td>
      </tr>
    </tbody>
  </table>
  </div>




  ```python
  test_inverse = DFE.inverse_transform(test, ["Embarked", "Sex"])
  ```


  ```python
  test_inverse.head()
  ```




  <div>
  <style scoped>
      .dataframe tbody tr th:only-of-type {
          vertical-align: middle;
      }

      .dataframe tbody tr th {
          vertical-align: top;
      }

      .dataframe thead th {
          text-align: right;
      }
  </style>
  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: right;">
        <th></th>
        <th>Survived</th>
        <th>Pclass</th>
        <th>Age</th>
        <th>SibSp</th>
        <th>Parch</th>
        <th>Fare</th>
        <th>Famsize</th>
        <th>Embarked</th>
        <th>Sex</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>0</th>
        <td>0</td>
        <td>3</td>
        <td>22.0</td>
        <td>1</td>
        <td>0</td>
        <td>7.2500</td>
        <td>2</td>
        <td>S</td>
        <td>male</td>
      </tr>
      <tr>
        <th>1</th>
        <td>1</td>
        <td>1</td>
        <td>38.0</td>
        <td>1</td>
        <td>0</td>
        <td>71.2833</td>
        <td>2</td>
        <td>C</td>
        <td>female</td>
      </tr>
      <tr>
        <th>2</th>
        <td>1</td>
        <td>3</td>
        <td>26.0</td>
        <td>0</td>
        <td>0</td>
        <td>7.9250</td>
        <td>1</td>
        <td>S</td>
        <td>female</td>
      </tr>
      <tr>
        <th>3</th>
        <td>1</td>
        <td>1</td>
        <td>35.0</td>
        <td>1</td>
        <td>0</td>
        <td>53.1000</td>
        <td>2</td>
        <td>S</td>
        <td>female</td>
      </tr>
      <tr>
        <th>4</th>
        <td>0</td>
        <td>3</td>
        <td>35.0</td>
        <td>0</td>
        <td>0</td>
        <td>8.0500</td>
        <td>1</td>
        <td>S</td>
        <td>male</td>
      </tr>
    </tbody>
  </table>
  </div>
