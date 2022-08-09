import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler




def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing.sort_values(by='num_rows_missing', ascending=False)



def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})
    rows_missing = df.merge(rows_missing,
                        left_index=True,
                        right_index=True)[['num_cols_missing', 'percent_cols_missing']]
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)


def summarize(df):
    '''
    This function will take in a single argument (a pandas dataframe) and 
    output to console various statistices on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # value_counts()
    # observation of nulls in the dataframe
    '''
    print('----------------------')
    print('Dataframe head')
    print(df.head(3))
    print('----------------------')
    print('Dataframe Info ')
    print(df.info())
    print('----------------------')
    print('Dataframe Description')
    print(df.describe())
    print('----------------------')
    num_cols = [col for col in df.columns if df[col].dtypes != 'object']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('----------------------')
    print('Dataframe value counts ')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts())
        else:
            # define bins for continuous columns and don't sort them
            print(df[col].value_counts(bins=10, sort=False))
    print('----------------------')
    print('nulls in df by column')
    print(nulls_by_col(df))
    print('----------------------')
    print('null in df by row')
    print(nulls_by_row(df))
    print('----------------------')



def remove_columns(df, cols_to_remove):
    df = df.drop(columns=cols_to_remove)
    return df


def series_upper_outliers(s, k=1.5):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))


def df_upper_outliers(df, k, cols):
    for col in cols:
        q1, q3 = df[col].quantile([.25, 0.75])
        iqr = q3 - q1
        upper_bound = q3 + k * iqr
    return df.apply(lambda x: max([x - upper_bound, 0]))

def df_lower_outliers(df, k, cols):
    for col in cols:
        q1, q3 = df[col].quantile([.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
    return df.apply(lambda x: max([x - lower_bound, 0]))    



def remove_outliers(df, k, cols):
    # df = df_upper_outliers()
    # df = df_lower_outliers()

     # return dataframe without outliers
    for col in cols:
        q1, q3 = df[col].quantile([.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr


        df = df[(df[f'{col}'] > lower_bound) & (df[f'{col}'] < upper_bound)]
    return df    
    
def encode_rows(df):
     dummies=pd.get_dummies(df['gender'], dummy_na=False, 
                                drop_first=True)

    # rename columns that have been one hot encoded
     dummies = dummies.rename(columns={'Male': 'is_male'})  

    # join dummy df to original df
     df = pd.concat([df, dummies], axis=1)

    # drop encoded column
     df = df.drop(['gender'], axis=1)
    
     return df


def add_upper_outlier_columns(df, k=1.5):
    '''
    Add a column with the suffix _outliers for all the numeric columns
    in the given dataframe.
    '''
    for col in df.select_dtypes('float64'):
        df[col + '_outliers_upper'] = df_upper_outliers(df[col], k)
    return df


def handle_missing_values(df, prop_required_columns=0.5, prop_required_row=0.75):
    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold) #1, or ‘columns’ : Drop columns which contain missing value
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold) #0, or ‘index’ : Drop rows which contain missing values.
    return df

def data_prep(df, cols_to_remove=[], prop_required_column=0.5, prop_required_row=0.75):
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df


def split_data(df):

    # split the data
    train_validate, test = train_test_split(df, test_size=.2, 
                                            random_state=123)
    train, validate = train_test_split(train_validate, test_size=.2, 
                                       random_state=123)
    return train, validate, test   



def MinMax_scaler(x_train, x_validate, x_test):
 
    scaler = MinMaxScaler().fit(x_train)

    scaler.fit(x_train)
    
    x_train_scaled = pd.DataFrame(scaler.transform(x_train), index=x_train.index, columns=x_train.columns)
    x_validate_scaled = pd.DataFrame(scaler.transform(x_validate), index=x_validate.index, columns=x_validate.columns)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), index=x_test.index, columns = x_test.columns)
    
    return x_train_scaled, x_validate_scaled, x_test_scaled

