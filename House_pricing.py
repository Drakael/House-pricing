import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import MSAI_SK_Solver
from sklearn.ensemble import GradientBoostingRegressor


# fonction utile pour le tracing
def p(mess, obj):
    """Useful function for tracing"""
    if hasattr(obj, 'shape'):
        print(mess, type(obj), obj.shape, "\n", obj)
    else:
        print(mess, type(obj), "\n", obj)


plt.close('all')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


predict_column = 'SalePrice'

solver = MSAI_SK_Solver.MSAI_SK_Solver()

# solver.null_report(train)

nan_NA = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
          'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
          'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature',
          'MSZoning']
zeros = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
         'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars',
         'GarageArea']


def clean_lot_frontage(df):
    for i, row in df.iterrows():
        if np.isnan(row['LotFrontage']):
            if row['LotConfig'] != 'Inside':
                df.at[i, 'LotFrontage'] = df['LotFrontage'].median()
            else:
                df.at[i, 'LotFrontage'] = 0
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


mappings = {}
mappings['Alley'] = {
    'values': {
        'Pave': 'Paved',
        'Grvl': 'Gravel',
        'NA': 'No access'
    },
    'type': 'num'
}
mappings['BldgType'] = {
    'values': {
        '1Fam': 'Single-family Detached',
        'TwnhsE': 'Townhouse End Unit',
        'Twnhs': 'Townhouse Inside Unit',
        'Duplex': 'Duplex',
        '2fmCon': 'Two-family Conversion; originally\
                   built as one-family dwelling'
    },
    'type': 'num'
}
mappings['BsmtCond'] = {
    'values': {
        'Ex': 'Excellent',
        'Gd': 'Good',
        'TA': 'Typical - slight dampness allowed',
        'NA': 'No Basement',
        'Fa': 'Fair - dampness or some cracking or settling',
        'Po': 'Poor - Severe cracking, settling, or wetness'
    },
    'type': 'num'
}
mappings['BsmtExposure'] = {
    'values': {
        'Gd': 'Good Exposure',
        'Av': 'Average Exposure',
        'No': 'No Exposure',
        'Mn': 'Mimimum Exposure',
        'NA': 'No Basement'
    },
    'type': 'num'
}
mappings['BsmtFinType1'] = {
    'values': {
        'GLQ': 'Good Living Quarters',
        'Unf': 'Unfinished',
        'ALQ': 'Average Living Quarters',
        'Rec': 'Average Rec Room',
        'LwQ': 'Low Quality',
        'BLQ': 'Below Average Living Quarters',
        'NA': 'No Basement'
    },
    'type': 'num'
}
mappings['BsmtFinType2'] = {
    'values': {
        'Unf': 'Unfinished',
        'ALQ': 'Average Living Quarters',
        'GLQ': 'Good Living Quarters',
        'Rec': 'Average Rec Room',
        'LwQ': 'Low Quality',
        'BLQ': 'Below Average Living Quarters',
        'NA': 'No Basement'
    },
    'type': 'num'
}
mappings['BsmtQual'] = {
    'values': {
        'Ex': 'Excellent (100+ inches)',
        'Gd': 'Good (90-99 inches)',
        'TA': 'Typical (80-89 inches)',
        'Fa': 'Fair (70-79 inches)',
        'Po': 'Poor (<70 inches)',
        'NA': 'No Basement'
    },
    'type': 'num'
}
mappings['CentralAir'] = {
    'values': {
        'Y': 'Yes',
        'N': 'No'
    },
    'type': 'num'
}
mappings['Condition1'] = {
    'values': {
        'Norm': 'Normal',
        'RRNn': 'Within 200\' of North-South Railroad',
        'RRAn': 'Adjacent to North-South Railroad',
        'PosN': 'Near positive off-site feature--park, greenbelt, etc.',
        'PosA': 'Adjacent to postive off-site feature',
        'RRNe': 'Within 200\' of East-West Railroad',
        'RRAe': 'Adjacent to East-West Railroad',
        'Feedr': 'Adjacent to feeder street',
        'Artery': 'Adjacent to arterial street'
    },
    'type': 'num'
}
mappings['Condition2'] = {
    'values': {
        'Artery': 'Adjacent to arterial street',
        'Feedr': 'Adjacent to feeder street',
        'Norm': 'Normal',
        'RRNn': 'Within 200\' of North-South Railroad',
        'RRAn': 'Adjacent to North-South Railroad',
        'PosN': 'Near positive off-site feature--park, greenbelt, etc.',
        'PosA': 'Adjacent to postive off-site feature',
        'RRNe': 'Within 200\' of East-West Railroad',
        'RRAe': 'Adjacent to East-West Railroad'
    },
    'type': 'num'
}
mappings['Electrical'] = {
    'values': {
        'SBrkr': 'Standard Circuit Breakers & Romex',
        'FuseA': 'Fuse Box over 60 AMP and all Romex wiring (Average)',
        'FuseF': '60 AMP Fuse Box and mostly Romex wiring (Fair)',
        'Mix': 'Mixed',
        'FuseP': '60 AMP Fuse Box and mostly knob & tube wiring (poor)'
    },
    'type': 'num'
}
mappings['ExterCond'] = {
    'values': {
        'Ex': 'Excellent',
        'Gd': 'Good',
        'TA': 'Average/Typical',
        'Fa': 'Fair',
        'Po': 'Poor'
    },
    'type': 'num'
}
mappings['ExterQual'] = {
    'values': {
        'Ex': 'Excellent',
        'Gd': 'Good',
        'TA': 'Average/Typical',
        'Fa': 'Fair'
    },
    'type': 'num'
}
mappings['Exterior1st'] = {
    'values': {
        'ImStucc': 'Imitation Stucco',
        'Stone': 'Stone',
        'VinylSd': 'Vinyl Siding',
        'Wd Sdng': 'Wood Siding',
        'CemntBd': 'Cement Board',
        'BrkFace': 'Brick Face',
        'MetalSd': 'Metal Siding',
        'Plywood': 'Plywood',
        'HdBoard': 'Hard Board',
        'Stucco': 'Stucco',
        'WdShing': 'Wood Shingles',
        'AsbShng': 'Asbestos Shingles',
        'CBlock': 'Cinder Block',
        'AsphShn': 'Asphalt Shingles',
        'BrkComm': 'Brick Common',
        'Other': 'Other',
        'PreCast': 'PreCast'
    },
    'type': 'num'
}
mappings['Exterior2nd'] = {
    'values': {
        'VinylSd': 'Vinyl Siding',
        'CmentBd': 'Cement Board',
        'BrkFace': 'Brick Face',
        'MetalSd': 'Metal Siding',
        'Wd Sdng': 'Wood Siding',
        'Plywood': 'Plywood',
        'HdBoard': 'Hard Board',
        'ImStucc': 'Imitation Stucco',
        'Other': 'Other',
        'Stucco': 'Stucco',
        'Wd Shng': 'Wood Shingles',
        'Stone': 'Stone',
        'AsphShn': 'Asphalt Shingles',
        'CBlock': 'Cinder Block',
        'AsbShng': 'Asbestos Shingles',
        'Brk Cmn': 'Brick Common',
        'PreCast': 'PreCast'
    },
    'type': 'num'
}
mappings['Fence'] = {
    'values': {
        'NA': 'No Fence',
        'GdWo': 'Good Wood',
        'GdPrv': 'Good Privacy',
        'MnPrv': 'Minimum Privacy',
        'MnWw': 'Minimum Wood/Wire'
    },
    'type': 'num'
}
mappings['FireplaceQu'] = {
    'values': {
        'Ex': 'Excellent - Exceptional Masonry Fireplace',
        'Gd': 'Good - Masonry Fireplace in main level',
        'TA': 'Average - Prefabricated Fireplace in main living\
               area or Masonry Fireplace in basement',
        'Fa': 'Fair - Prefabricated Fireplace in basement',
        'Po': 'Poor - Ben Franklin Stove',
        'NA': 'No Fireplace'
    },
    'type': 'num'
}
mappings['Foundation'] = {
    'values': {
        'PConc': 'Poured Contrete',
        'CBlock': 'Cinder Block',
        'Wood': 'Wood',
        'Stone': 'Stone',
        'Slab': 'Slab',
        'BrkTil': 'Brick & Tile'
    },
    'type': 'num'
}
mappings['Functional'] = {
    'values': {
        'Typ': 'Typical Functionality',
        'Maj1': 'Major Deductions 1',
        'Min1': 'Minor Deductions 1',
        'Min2': 'Minor Deductions 2',
        'Mod': 'Moderate Deductions',
        'Maj2': 'Major Deductions 2',
        'Sal': 'Salvage only',
        'Sev': 'Severely Damaged'
    },
    'type': 'num'
}
mappings['GarageCond'] = {
    'values': {
        'TA': 'Typical/Average',
        'Gd': 'Good',
        'Ex': 'Excellent',
        'Fa': 'Fair',
        'Po': 'Poor',
        'NA': 'No Garage'
    },
    'type': 'num'
}
mappings['GarageFinish'] = {
    'values': {
        'Fin': 'Finished',
        'RFn': 'Rough Finished',
        'Unf': 'Unfinished',
        'NA': 'No Garage'
    },
    'type': 'num'
}
mappings['GarageQual'] = {
    'values': {
        'Ex': 'Excellent',
        'Gd': 'Good',
        'TA': 'Typical/Average',
        'Fa': 'Fair',
        'Po': 'Poor',
        'NA': 'No Garage'
    },
    'type': 'num'
}
mappings['GarageType'] = {
    'values': {
        'BuiltIn': 'Built-In (Garage part of house -\
                    typically has room above garage)',
        'Attchd': 'Attached to home',
        'Detchd': 'Detached from home',
        'Basment': 'Basement Garage',
        '2Types': 'More than one type of garage',
        'CarPort': 'Car Port',
        'NA': 'No Garage'
    },
    'type': 'num'
}
mappings['Heating'] = {
    'values': {
        'GasA': 'Gas forced warm air furnace',
        'GasW': 'Gas hot water or steam heat',
        'OthW': 'Hot water or steam heat other than gas',
        'Grav': 'Gravity furnace',
        'Wall': 'Wall furnace',
        'Floor': 'Floor Furnace'
    },
    'type': 'num'
}
mappings['HeatingQC'] = {
    'values': {
        'Ex': 'Excellent',
        'Gd': 'Good',
        'TA': 'Typical/Average',
        'Fa': 'Fair',
        'Po': 'Poor'
    },
    'type': 'num'
}
mappings['HouseStyle'] = {
    'values': {
        '2Story': 'Two story',
        '1Story': 'One story',
        '1.5Fin': 'One and one-half story: 2nd level finished',
        '2.5Fin': 'Two and one-half story: 2nd level finished',
        '2.5Unf': 'Two and one-half story: 2nd level unfinished',
        'SLvl': 'Split Level',
        'SFoyer': 'Split Foyer',
        '1.5Unf': 'One and one-half story: 2nd level unfinished'
    },
    'type': 'num'
}
mappings['KitchenQual'] = {
    'values': {
        'Ex': 'Excellent',
        'Gd': 'Good',
        'TA': 'Typical/Average',
        'Fa': 'Fair',
        'Po': 'Poor'
    },
    'type': 'num'
}
mappings['LandContour'] = {
    'values': {
        'Lvl': 'Near Flat/Level',
        'HLS': 'Hillside - Significant slope from side to side',
        'Low': 'Depression',
        'Bnk': 'Banked - Quick and significant rise from street\
                grade to building'
    },
    'type': 'num'
}
mappings['LandSlope'] = {
    'values': {
        'Gtl': 'Gentle slope',
        'Mod': 'Moderate Slope',
        'Sev': 'Severe Slope'
    },
    'type': 'num'
}
mappings['LotConfig'] = {
    'values': {
        'Inside': 'Inside lot',
        'CulDSac': 'Cul-de-sac',
        'Corner': 'Corner lot',
        'FR2': 'Frontage on 2 sides of property',
        'FR3': 'Frontage on 3 sides of property'
    },
    'type': 'num'
}
mappings['LotShape'] = {
    'values': {
        'Reg': 'Regular',
        'IR1': 'Slightly irregular',
        'IR2': 'Moderately Irregular',
        'IR3': 'Irregular'
    },
    'type': 'num'
}
mappings['MSSubClass'] = {
    'values': {
        '80': 'SPLIT OR MULTI-LEVEL',
        '180': 'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER',
        '90': 'DUPLEX - ALL STYLES AND AGES',
        '190': '2 FAMILY CONVERSION - ALL STYLES AND AGES',
        '120': '1-STORY PUD (Planned Unit Development) - 1946 & NEWER',
        '40': '1-STORY W/FINISHED ATTIC ALL AGES',
        '45': '1-1/2 STORY - UNFINISHED ALL AGES',
        '50': '1-1/2 STORY FINISHED ALL AGES',
        '60': '2-STORY 1946 & NEWER',
        '70': '2-STORY 1945 & OLDER',
        '75': '2-1/2 STORY ALL AGES',
        '85': 'SPLIT FOYER',
        '150': '1-1/2 STORY PUD - ALL AGES',
        '160': '2-STORY PUD - 1946 & NEWER',
        '20': '1-STORY 1946 & NEWER ALL STYLES',
        '30': '1-STORY 1945 & OLDER'
    },
    'type': 'num'
}
mappings['MSZoning'] = {
    'values': {
        'RL': 'Residential Low density',
        'FV': 'Floating Village Residential',
        'RM': 'Residential Medium Density',
        'RH': 'Residential High Density',
        'C (all)': 'Commercial',
        'NA': 'Dont know'
    },
    'type': 'num'
}
mappings['MasVnrType'] = {
    'values': {
        'Stone': 'Stone',
        'BrkFace': 'Brick Face',
        'None': 'None',
        'BrkCmn': 'Brick Common',
        'CBlock': 'Cinder Block'
    },
    'type': 'num'
}
mappings['MiscFeature'] = {
    'values': {
        'Elev': 'Elevator',
        'Gar2': '2nd Garage (if not described in garage section)',
        'Shed': 'Shed (over 100 SF)',
        'TenC': 'Tennis Court',
        'Othr': 'Other',
        'NA': 'None'
    },
    'type': 'num'
}
mappings['Neighborhood'] = {
    'values': {
        'NoRidge': 'Northridge',
        'NridgHt': 'Northridge Heights',
        'StoneBr': 'Stone Brook',
        'Timber': 'Timberland',
        'Veenker': 'Veenker',
        'Somerst': 'Somerset',
        'ClearCr': 'Clear Creek',
        'Crawfor': 'Crawford',
        'CollgCr': 'College Creek',
        'Blmngtn': 'Bloomington Heights',
        'Gilbert': 'Gilbert',
        'NWAmes': 'Northwest Ames',
        'SawyerW': 'Sawyer West',
        'Mitchel': 'Mitchell',
        'NAmes': 'North Ames',
        'NPkVill': 'Northpark Villa',
        'SWISU': 'South & West of Iowa State University',
        'Blueste': 'Bluestem',
        'Sawyer': 'Sawyer',
        'OldTown': 'Old Town',
        'Edwards': 'Edwards',
        'BrkSide': 'Brookside',
        'BrDale': 'Briardale',
        'IDOTRR': 'Iowa DOT and Rail Road',
        'MeadowV': 'Meadow Village'
    },
    'type': 'num'
}
mappings['PavedDrive'] = {
    'values': {
        'Y': 'Paved',
        'P': 'Partial Pavement',
        'N': 'Dirt/Gravel'
    },
    'type': 'num'
}
mappings['PoolQC'] = {
    'values': {
        'Ex': 'Excellent',
        'Gd': 'Good',
        'Fa': 'Fair',
        'NA': 'No Pool'
    },
    'type': 'num'
}
mappings['RoofMatl'] = {
    'values': {
        'CompShg': 'Standard (Composite) Shingle',
        'WdShake': 'Wood Shakes',
        'WdShngl': 'Wood Shingles',
        'Tar&Grv': 'Gravel & Tar',
        'Metal': 'Metal',
        'Roll': 'Roll',
        'ClyTile': 'Clay or Tile',
        'Membran': 'Membrane'
    },
    'type': 'num'
}
mappings['RoofStyle'] = {
    'values': {
        'Hip': 'Hip',
        'Gable': 'Gable',
        'Shed': 'Shed',
        'Mansard': 'Mansard',
        'Flat': 'Flat',
        'Gambrel': 'Gabrel (Barn)'
    },
    'type': 'num'
}
mappings['SaleCondition'] = {
    'values': {
        'Partial': 'Home was not completed when last assessed\
                    (associated with New Homes)',
        'Normal': 'Normal Sale',
        'Alloca': 'Allocation - two linked properties with separate\
                   deeds, typically condo with a garage unit',
        'Abnorml': 'Abnormal Sale -  trade, foreclosure, short sale',
        'Family': 'Sale between family members',
        'AdjLand': 'Adjoining Land Purchase'
    },
    'type': 'num'
}
mappings['SaleType'] = {
    'values': {
        'New': 'Home just constructed and sold',
        'WD': 'Warranty Deed - Conventional',
        'ConLI': 'Contract Low Interest',
        'Con': 'Contract 15% Down payment regular terms',
        'CWD': 'Warranty Deed - Cash',
        'ConLD': 'Contract Low Down',
        'COD': 'Court Officer Deed/Estate',
        'ConLw': 'Contract Low Down payment and low interest',
        'Oth': 'Other',
        'VWD': 'Warranty Deed - VA Loan'
    },
    'type': 'num'
}
mappings['Street'] = {
    'values': {
        'Pave': 'Paved',
        'Grvl': 'Gravel'
    },
    'type': 'num'
}
mappings['Utilities'] = {
    'values': {
        'AllPub': 'All public Utilities (E,G,W,& S)',
        'NoSeWa': 'Electricity and Gas Only'
    },
    'type': 'num'
}

train = clean_data(train)
test = clean_data(test)

print('len(train)', len(train))
# solver.null_report(train)

# train, mapping = solver.auto_clean(train, predict_column)

train = solver.map_clean2(train, mappings)
test = solver.map_clean2(test, mappings)

# retrait d'outliers
train = train[train['GrLivArea'] < 4600]
train = train[train['GarageArea'] < 1200]
# train = train[train['TotalBsmtSF'] < 6000]
# train = train[train['1stFlrSF'] < 4000]
train = train[train['SalePrice'] < 600000]
train = train[train['TotRmsAbvGrd'] < 14]
train = train[train['YearBuilt'] > 1895]


# p('train',train)
# solver.null_report(train)

# solver.fit_all().fit_best()

# solver.feed_data(train, predict_column, 78, 'Id')
# acc, mean, std = solver.fit(GradientBoostingRegressor(),0.1)

train_accuracy = []
mean_outcome = []
std_outcome = []
for i in range(53, 58):
    solver.feed_data(train, predict_column, i, 'Id')
    acc, mean, std = solver.fit(GradientBoostingRegressor(loss='ls'))
    train_accuracy.append(acc)
    mean_outcome.append(mean)
    std_outcome.append(std)
    solver.dump_result(test, 'test_output_dump_'+str(i)+'_'+"{0:.6f}".format(solver.accuracy)+'.csv')

plt.figure()
plt.plot(train_accuracy, color='blue')
plt.plot(mean_outcome, color='red')
plt.plot(std_outcome, color='green')
plt.show()

# solver.feed_data(train, predict_column, 65, 'Id')
# solver.fit(GradientBoostingRegressor(),0.1)
# solver.dump_result(test, 'test_output.csv')

# solver.get_correlations(train, predict_column)

# solver.first_analysis(train, predict_column)

# solver.plot_most_important(train, predict_column, nb=18, start=1)
