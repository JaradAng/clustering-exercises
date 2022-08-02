import pandas as pd
import numpy as np
import os
from env import get_db_url

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#Function to import the SQL database into jupyter notebook
def zillow_data():
    filename = 'zillow.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        zillow_df = pd.read_sql('''SELECT 
    *
FROM
    properties_2017
        LEFT JOIN # maybe right join this one
    predictions_2017 AS pred USING (parcelid)
        LEFT JOIN
    airconditioningtype USING (airconditioningtypeid)
        LEFT JOIN
    architecturalstyletype USING (architecturalstyletypeid)
        LEFT JOIN
    buildingclasstype USING (buildingclasstypeid)
        LEFT JOIN
    heatingorsystemtype USING (heatingorsystemtypeid)
        LEFT JOIN
    typeconstructiontype USING (typeconstructiontypeid)
WHERE
    pred.transactiondate LIKE '2017%%'
        AND latitude IS NOT NULL
        AND longitude IS NOT NULL;''', get_db_url('zillow'))
        # zillow_df.to_csv(filename)

        return zillow_df





def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # .value_counts()
    # observation of nulls in the dataframe
    '''
    print('SUMMARY REPORT')
    print('=====================================================\n\n')
    print('Dataframe head: ')
    print(df.head(3))
    print('=====================================================\n\n')
    print('Dataframe info: ')
    print(df.info())
    print('=====================================================\n\n')
    print('Dataframe Description: ')
    print(df.describe())
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('=====================================================')
    print('DataFrame value counts: ')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts(), '\n')
        else:
            print(df[col].value_counts(bins=10, sort=False), '\n')
    print('=====================================================')
    print('nulls in dataframe by column: ')
    print(nulls_by_col(df))
    print('=====================================================')
    print('nulls in dataframe by row: ')
    print(nulls_by_row(df))
    print('=====================================================')


def trim_bad_data_zillow(df):
    # If it's not single unit, it's not a single family home.
    df = df[~(df.unitcnt > 1)]
    # If the lot size is smaller than the finished square feet, it's probably bad data or not a single family home
    df = df[~(df.lotsizesquarefeet < df.calculatedfinishedsquarefeet)]
    # If the finished square feet is less than 500 it is likeley an apartment, or bad data
    df = df[~(df.calculatedfinishedsquarefeet < 500)]
    # If there are no bedrooms, likely a loft or bad data
    df = df[~(df.bedroomcnt < 1)]
    # Drop duplicate parcels
    df = df.drop_duplicates(subset='parcelid')
    return df






def clean_zillow(df):

    #Making pools boolean despite the amount of nulls, if a house has a pool it will be listed in the features becuase it is a high ticket item
    df['poolcnt'] = np.where((df['poolcnt'] == 1.0) , True , False)
    
    # Assigning the value of the car garage to the dataset if its above 1 and making the nulls to 0 doing this because garages are important enough to list and there are as many nulls with garage sq
    df['garagecarcnt'] = np.where((df['garagecarcnt'] >= 1.0) , df['garagecarcnt'] , 0)  


    #Drop the columns with null values because they only make up about 0.5% and would have less impact on model than imputing value
    df = df.dropna()
    
    # Rename Columns and assigning data type if needed
    df["fed_code"] = df["fips"].astype(int)
    df["year_built"] = df["yearbuilt"].astype(int)
    df["beds"] = df["bedroomcnt"].astype(int)    
    df["home_value"] = df["taxvaluedollarcnt"].astype(float)
    df["sq_ft"] = df["calculatedfinishedsquarefeet"].astype(float)
    df["baths"] = df["bathroomcnt"]
    df["lot_size"] = df["lotsizesquarefeet"]
    df["pools"] = df["poolcnt"]
    df["garages"] = df["garagecarcnt"]

    return df
  
def feature_engineering(df):
    #Feature engineering new variables to combat multicolinearty and test to see if new features help the model

    df['pool_encoded'] = df.pools.map({True:1, False:0})
    
    # I will drop bed and bath in exchange for the ratio which accomplishes the same task without multicollinearity
    df['bed_bath_ratio'] = df['beds'] / df['baths']
    # Leaving in the two original columns and seeing if the overall sq ft might make a difference
    df['overall_size'] = df['sq_ft'] + df['lot_size']
    #dropping year built and turning it into house age which will then be scaled
    df['house_age'] = 2017 - df['year_built']
    
    # making dummies and encoded values to help machine learning
    # dummy_df = pd.get_dummies(df[['garages']], dummy_na=False,drop_first=False)

    # df = pd.concat([df, dummy_df], axis=1)


    #Deleting duplicate rows
    df = df.drop(columns=['fips', 'yearbuilt', 'bedroomcnt', 'taxvaluedollarcnt', 'calculatedfinishedsquarefeet', 'bathroomcnt', 'poolcnt', 'lotsizesquarefeet', 'year_built', 'garagecarcnt', 'beds', 'baths'])
    
    return df







# def handle_outliers(df):
#     """Manually handle outliers that do not represent properties likely for 99% of buyers and zillow visitors"""
#     df = df[df.baths <= 6]

#     df =df[df.baths > 0]
    
#     df = df[df.beds <= 6]

#     df =df[df.beds > 0]

#     df = df[df.sq_ft <= 7_000]

#     df = df[df.sq_ft > 700]

#     df = df[df.home_value < 1_500_000]

#     df = df[df.lot_size <=100_000]

#     df = df[df.garages <=5]

#     return df



def wrangle_zillow():

    df = zillow_data()

    df = clean_zillow(df)

    # df = handle_outliers(df)

    df = feature_engineering(df)

    df.to_csv("zillow.csv", index=False)

    return df

# split the data before fitting scalers 
def split_zillow(df):

    # split the data
    train_validate, test = train_test_split(df, test_size=.2, 
                                            random_state=123)
    train, validate = train_test_split(train_validate, test_size=.2, 
                                       random_state=123)
    return train, validate, test    

# cols_to_scale = ['sq_ft', 'overall_size', 'house_age', 'garages', 'lot_size']

# Function applying the min max scaled to the the x variables
def MinMax_scaler(x_train, x_validate, x_test):
 
    scaler = MinMaxScaler().fit(x_train)

    scaler.fit(x_train)
    
    x_train_scaled = pd.DataFrame(scaler.transform(x_train), index=x_train.index, columns=x_train.columns)
    x_validate_scaled = pd.DataFrame(scaler.transform(x_validate), index=x_validate.index, columns=x_validate.columns)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), index=x_test.index, columns = x_test.columns)
    
    return x_train_scaled, x_validate_scaled, x_test_scaled
    
