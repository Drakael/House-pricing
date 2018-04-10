import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import MSIA_SK_Solver

#fonction utile pour le tracing
def p(mess,obj):
    """Useful function for tracing"""
    if hasattr(obj,'shape'):
        print(mess,type(obj),obj.shape,"\n",obj)
    else:
        print(mess,type(obj),"\n",obj)
        

plt.close('all')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


predict_column = 'SalePrice'

solver = MSIA_SK_Solver.MSIA_SK_Solver()


#solver.null_report(train)

nan_NA = ['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature','MSZoning']
zeros = ['MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea']

def clean_lot_frontage(df):
    for i, row in df.iterrows():
        if np.isnan(row['LotFrontage']) == True:
            df.at[i,'LotFrontage'] = df['LotFrontage'].median() if row['LotConfig'] != 'Inside' else 0
    return df

def clean_nan_saletype(df):
    df['SaleType'] = df['SaleType'].fillna('Oth')
    return df

def clean_nan_masvnrtype(df):
    df['MasVnrType'] = df['MasVnrType'].fillna('None')
    return df

def clean_nan_electrical(df):
    df['Electrical'] = df['Electrical'].fillna('Mix')
    return df

def clean_nan_exterior(df):
    df['Exterior1st'] = df['Exterior1st'].fillna('Other')
    df['Exterior2nd'] = df['Exterior2nd'].fillna('Other')
    return df

def clean_nan_kitchen_quality(df):
    df['KitchenQual'] = df['KitchenQual'].fillna('TA')
    return df

def clean_nan_functional(df):
    df['Functional'] = df['Functional'].fillna('Mod')
    return df

def clean_nan_utilities(df):
    df['Utilities'] = df['Utilities'].fillna('AllPub')
    return df

def clean_data(df):
    df = clean_lot_frontage(df)
    df = clean_nan_saletype(df)
    df = clean_nan_masvnrtype(df)
    df = clean_nan_electrical(df)
    df = clean_nan_exterior(df)
    df = clean_nan_kitchen_quality(df)
    df = clean_nan_functional(df)
    df = clean_nan_utilities(df)
    for col in nan_NA:
        df[col] = df[col].fillna('NA')
    for zer in zeros:
        df[zer] = df[zer].fillna(0)
    return df

train = clean_data(train)

train, mapping = solver.auto_clean(train, predict_column)

test = solver.map_clean(test, mapping)

#solver.get_correlations(train, predict_column)

#solver.first_analysis(train, predict_column)

#solver.plot_most_important(train, predict_column, nb=18, start=1)

