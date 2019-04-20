
# Libraries
import numpy as np
import pandas as pd


#region IMPORTS

# Import Excel Spreadsheet
df = pd.read_excel("file.xlsx", "spreadsheet")
# Import several spreadsheets
filenames = ['spread1', 'spread2', 'spread3']
dataframes = []
for i in filenames:
    dataframes.append(pd.read_excel("file.xlsx", i))

# Import csv file
df = pr.read_csv("file.csv")
# Import several csv files
filenames = ['file1.csv', 'file2.csv', 'file3.csv']
dataframes = []
for i in filenames:
    dataframes.append(pd.read_csv(i))

# Import from SQL
from sqlalchemy import create_engine
engine = create_engine('sqlite:///:memory:')
df = pd.read_sql("SELECT x FROM my_table;", engine)
df = pd.read_sql_table('my_table', engine)
df = pd.read_sql_query("SELECT x, y FROM my_table;", engine)

# Export to csv
df.to_csv("df.csv", sep="\t")

# From Lists to Dicts to DataFrames
zipped = list(zip(list_keys, list_values))
# dictionary
dix = dict(zipped)
# df
df = pd.DataFrame(dix)

#endregion




#region DF INSPECTION

# General enquiries
df.shape
df.columns
df.describe()
df.info()

# Subset ROWS
df.head()
df.tail()
df.iloc[10:20] # select rows by position
df[df.x>100]  # select rows that meet logical condition
df[df.x!="ABCD"]

# subset COLUMNS
df.loc[:, "x1":"x4"]  # select all rows, columns x1 to x4 (incl.!)
df.iloc[:, [1, 2, 5]]  # select all rows, col. 1, 2, 5 (zero indexed)

# subset ROWS and COLUMNS
df.loc[df.x1>100, ["x2", "x3"]]  # select rows meeting cond., col. x2, x3

# Filter Joins
df1[df1.x1.isin(df2.x2)].shape[0]  # number of x1-values contained in x2
df1[~df1.x1.isin(df2.x2)].shape[0]  # number of x2-values NOT contained in x2
df2[df2.x2.isin(df1.x1)].shape[0]  # number of x2-values contained in x1
df2[~df2.x2.isin(df1.x1)].shape[0]  # number of x2-values NOT contained in x1

# Indizes
df.index
df.set_index("x1", inplace=True)
df.reset_index(inplace=True, drop=False)

# Multiindexes, hierarchical
df = df.set_index(["state", "month"])  
df = df.sort_index()  # Sort the MultiIndex: df
NY_month1 = df.loc[("NY", 1), :]  # Look up data for NY in month 1: NY_month1
CA_TX_month2 = df.loc[(["CA", "TX"], 2), :]  # Look up data for CA and TX in month 2: CA_TX_month2
all_month2 = df.loc[(slice(None), 2), :]  # Look up data for all states in month 2: all_month2
byweekday = users.unstack(level="weekday")  # Unstack users by 'weekday': byweekday
print(byweekday.stack(level="weekday"))  # Stack byweekday by 'weekday' and print it
df = df.swaplevel(0, 1)  # Swap the levels of the index of df: df
newusers = newusers.sort_index()  # Sort the index of newusers: newusers


#endregion




#region VARIABLE TRANSFORMATION

# changing values in y of rows meeting x-value condition
df.loc[df["x"]=="NAM", "y"] = "NA"
# alternatively
df.y[df.x=="NAM"] = "NA"

# new df, adding suffixes to columns (e.g. to subs. merge)
df_suff = df.copy()
df_suff.columns = df_suff.columns.map(lambda x: str(x) + '_suff')

# recoding variable
df["x_recoded"] = np.where(
    df["x"]=='saved me money on international money transfers', 'Saver',
    (np.where(
        df["x"]=='both', 'Both',
        (np.where(
            df["x"]=='made me feel like I\'m part of a revolution', 'Revolutionary', 'None'
            )
        )
    ))
)

# replacing values
df.replace('\\0', np.NaN, inplace=True)  # replace "\0" with NaNs

# dropping observations with NAs
df.dropna(how="any").shape  # row dropped if ANY column contains NA
df.dropna(how="all").shape  # row dropped if ALL columns contain NAs
titanic.dropna(thresh=1000, axis='columns').info()  # drop columns with less than 1000 non-NA values

# taking logs
df["x"] = np.log(
    df.x.replace(0, 1))  # avoiding ln(0)=-inf

# datetime variable
df["end_timestamp"] = pd.Series(pd.date_range(start='20140331 12:00:00', end='20140331 12:00:00', periods=862))
df["t_delta"] = df.end_timestamp - df.user_create_timestamp
df['yrs_delta'] = df['t_delta'].apply(lambda x: float(x.days)/365)

# dummy variables 1
d1 = pd.get_dummies(df['x1'])
d1.columns = d1.columns.map(lambda x: str(x) + '_x1')
d2 = pd.get_dummies(df['x2'])
d2.columns = d2.columns.map(lambda x: str(x) + '_x2')
d3 = pd.get_dummies(df['x3'])
d3.columns = d3.columns.map(lambda x: str(x) + '_x3')

dummies = pd.concat([d1, d2, d3], axis=1)
dummies["<abcd_dummy"] = dummies['value1_x1'] + dummies['value2_x1']

# dummy and count variables
df['x_d'] = pd.notnull(df.x) * 1  # x is count var with nans
df['y_d'] = df.y>0  # y is timestamp
df.y_d = df.y_d * 1
df['x_n'] = df.x.fillna(0)  # x is count variable, nans = 0

# Vectorized operations 1: .apply (apply fun to col-values)
# Write a function: to_celsius
def to_celsius(F):
    return 5/9*(F - 32)
df_celsius = weather[["Mean TemperatureF", "Mean Dew PointF"]].apply(to_celsius)

# Vectorized operations 2: .map (transform values acc. to dict look-up)
red_vs_blue = {"Obama": "blue", "Romney": "red"}
df['color'] = df.winner.map(red_vs_blue)

#endregion




#region DF RESHAPING


# Dropping Rows
df = df[df.x!="None"]  # Drop the n observations with "None" in x

# Concat dfs vertically
df = pd.concat([df1, df2])

# Concat dfs horizontally
df = pd.concat([df1, df2], axis=1)

# Pivot: More than one variable in y -> spread to columns
df1 = df.pivot(index="x", columns="y", values="z") 
df1.reset_index(inplace=True)
# Use a pivot table to display the count of each column: count_by_weekday1
count_by_weekday1 = users.pivot_table(index="weekday", aggfunc="count")

# Melt: values of one variable in columns -> bring into one column
df1 = df.melt(df, id_vars="x")
df1.reset_index(inplace=True)  # not sure, check for index

# Groupby: More than one observation per id_user
aggregations = {
    "x1": "count",
    "x2": "count",
    "x3": "sum",
    "x4": "sum"
    }
df1 = df.groupby("id_user").agg(aggregations) 
user_ref1.reset_index(inplace=True)

# more complex Groupby's (more ex. in 9_manipulating_dfs_pandas.py)
# .transform() -> element-wise application of fun
from scipy.stats import zscore
df_stand = df.groupby("region")[["life", "fertility"]].transform(zscore)
outliers = (df_stand['life'] < -3) | (df_stand["fertility"] > 3)
gm_outliers = df.loc[outliers, : ]



#endregion




#region DF MERGES

df1.set_index("x1", inplace=True)  # index
df2.set_index("y1", inplace=True)
df = pd.merge(df1, df2, how="left", left_index=True,  right_index=True)  # join
df.reset_index(inplace=True)  # deindex
df1.reset_index(inplace=True)
df2.reset_index(inplace=True)
df.rename(columns={'index':'x'}, inplace=True)  # rename former index-var

#endregion

