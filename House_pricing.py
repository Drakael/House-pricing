import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pydot
#import importlib


from matplotlib import cm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, explained_variance_score, confusion_matrix, mean_squared_error
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
#from sklearn.externals.six import StringIO
#importlib.import_module('MSIASolver')
import MSIASolver

#fonction utile pour le tracing
def p(mess,obj):
    """Useful function for tracing"""
    if hasattr(obj,'shape'):
        print(mess,type(obj),obj.shape,"\n",obj)
    else:
        print(mess,type(obj),"\n",obj)

def info_all(df):
    cnt = 0
    max_ = len(df.columns)
    while cnt<max_:
        cnt_init = cnt
        cnt+=15
        if cnt>max_:
            cnt=max_
        print(df.iloc[:,cnt_init:cnt].info()) 

def na_rows(df):
    return df.columns[df.isna().any()] 

def null_report(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print('Missing Data\n',missing_data.head(20))

def pca_analysis(df, n_comp = 20):
    print('\nRunning PCA ...')
    pca = PCA(n_components=n_comp, svd_solver='full', random_state=1001)
    df_pca = pca.fit_transform(df)
    p('df_pca',df_pca)
    print('Explained variance: %.4f' % pca.explained_variance_ratio_.sum())
    
    print('Individual variance contributions:')
    print('noise:',pca.noise_variance_)
    for j in range(pca.n_components_):
        print('ratio',pca.explained_variance_ratio_[j],'var',pca.explained_variance_[j],'val',pca.singular_values_[j],'mean',pca.mean_[j])
    print(np.mean(pca.get_covariance(),axis=1))
    print('columns',df.columns)

plt.close('all')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#sub = pd.read_csv('sample_submission.csv')


#train = train[:][10:60]

predict_column = 'SalePrice'

def get_correlations(df, predict_column, mess):
    #calcul et affichage des facteurs de corrélation des variables
    #methods = ['pearson', 'kendall', 'spearman']
    correlation = df.corr(method='pearson')
    #p('correlation',correlation)
    print("\n"+mess+' relatives = '+"\n",correlation[predict_column].sort_values(ascending= False),"\n")
    correlation = correlation[predict_column].abs().sort_values(ascending= False)
    print(mess+' absolues = ',type(correlation),"\n",correlation,"\n","\n")
    return correlation

get_correlations(train, predict_column, 'correlations initiales')

Id_train = train['Id'].tolist()
Id_test = test['Id'].tolist()
#Id_sub = sub['Id'].tolist()

mappings = {}
mappings['Alley'] = {
    'values':{
        'Pave':'Paved',
        'Grvl':'Gravel',
        'NA':'No access'
        },
    'type':'num_desc'
    }
mappings['BldgType'] = {
    'values':{
        '1Fam':'Single-family Detached',
        '2fmCon':'Two-family Conversion; originally built as one-family dwelling',
        'Duplex':'Duplex',
        'TwnhsE':'Townhouse End Unit',
        'Twnhs':'Townhouse Inside Unit'
        },
    'type':'num_desc'
    }
mappings['BsmtCond'] = {
    'values':{
        'Ex':'Excellent',
        'Gd':'Good',
        'TA':'Typical - slight dampness allowed',
        'Fa':'Fair - dampness or some cracking or settling',
        'Po':'Poor - Severe cracking, settling, or wetness',
        'NA':'No Basement'
        },
    'type':'num_desc'
    }
mappings['BsmtExposure'] = {
    'values':{
        'Gd':'Good Exposure',
        'Av':'Average Exposure',
        'Mn':'Mimimum Exposure',
        'No':'No Exposure',
        'NA':'No Basement'
        },
    'type':'num_desc'
    }
mappings['BsmtFinType1'] = {
    'values':{
        'GLQ':'Good Living Quarters',
        'ALQ':'Average Living Quarters',
        'BLQ':'Below Average Living Quarters',
        'Rec':'Average Rec Room',
        'LwQ':'Low Quality',
        'Unf':'Unfinished',
        'NA':'No Basement'
        },
    'type':'num_desc'
    }
mappings['BsmtFinType2'] = {
    'values':{
        'GLQ':'Good Living Quarters',
        'ALQ':'Average Living Quarters',
        'BLQ':'Below Average Living Quarters',
        'Rec':'Average Rec Room',
        'LwQ':'Low Quality',
        'Unf':'Unfinished',
        'NA':'No Basement'
        },
    'type':'num_desc'
    }
mappings['BsmtQual'] = {
    'values':{
        'Ex':'Excellent (100+ inches)',
        'Gd':'Good (90-99 inches)',
        'TA':'Typical (80-89 inches)',
        'Fa':'Fair (70-79 inches)',
        'Po':'Poor (<70 inches)',
        'NA':'No Basement'
        },
    'type':'num_desc'
    }
mappings['CentralAir'] = {
    'values':{
        'Y':'Yes',
        'N':'No'
        },
    'type':'num_desc'
    }
mappings['Condition1'] = {
    'values':{
        'Artery':'Adjacent to arterial street',
        'Feedr':'Adjacent to feeder street',
        'Norm':'Normal',
        'RRNn':'Within 200\' of North-South Railroad',
        'RRAn':'Adjacent to North-South Railroad',
        'PosN':'Near positive off-site feature--park, greenbelt, etc.',
        'PosA':'Adjacent to postive off-site feature',
        'RRNe':'Within 200\' of East-West Railroad',
        'RRAe':'Adjacent to East-West Railroad'
        },
    'type':'num_desc'
    }
mappings['Condition2'] = {
    'values':{
        'Artery':'Adjacent to arterial street',
        'Feedr':'Adjacent to feeder street',
        'Norm':'Normal',
        'RRNn':'Within 200\' of North-South Railroad',
        'RRAn':'Adjacent to North-South Railroad',
        'PosN':'Near positive off-site feature--park, greenbelt, etc.',
        'PosA':'Adjacent to postive off-site feature',
        'RRNe':'Within 200\' of East-West Railroad',
        'RRAe':'Adjacent to East-West Railroad'
        },
    'type':'num_desc'
    }
mappings['Electrical'] = {
    'values':{
        'SBrkr':'Standard Circuit Breakers & Romex',
        'FuseA':'Fuse Box over 60 AMP and all Romex wiring (Average)',
        'FuseF':'60 AMP Fuse Box and mostly Romex wiring (Fair)',
        'FuseP':'60 AMP Fuse Box and mostly knob & tube wiring (poor)',
        'Mix':'Mixed'
        },
    'type':'num_desc'
    }
mappings['ExterCond'] = {
    'values':{
        'Ex':'Excellent',
        'Gd':'Good',
        'TA':'Average/Typical',
        'Fa':'Fair',
        'Po':'Poor'
        },
    'type':'num_desc'
    }
mappings['ExterQual'] = {
    'values':{
        'Ex':'Excellent',
        'Gd':'Good',
        'TA':'Average/Typical',
        'Fa':'Fair'
        },
    'type':'num_desc'
    }
mappings['Exterior1st'] = {
    'values':{
        'AsbShng':'Asbestos Shingles',
        'AsphShn':'Asphalt Shingles',
        'BrkComm':'Brick Common',
        'BrkFace':'Brick Face',
        'CBlock':'Cinder Block',
        'CemntBd':'Cement Board',
        'HdBoard':'Hard Board',
        'ImStucc':'Imitation Stucco',
        'MetalSd':'Metal Siding',
        'Other':'Other',
        'Plywood':'Plywood',
        'PreCast':'PreCast',
        'Stone':'Stone',
        'Stucco':'Stucco',
        'VinylSd':'Vinyl Siding',
        'Wd Sdng':'Wood Siding',
        'WdShing':'Wood Shingles'
        },
    'type':'num_desc'
    }
mappings['Exterior2nd'] = {
    'values':{
        'AsbShng':'Asbestos Shingles',
        'AsphShn':'Asphalt Shingles',
        'Brk Cmn':'Brick Common',
        'BrkFace':'Brick Face',
        'CBlock':'Cinder Block',
        'CmentBd':'Cement Board',
        'HdBoard':'Hard Board',
        'ImStucc':'Imitation Stucco',
        'MetalSd':'Metal Siding',
        'Other':'Other',
        'Plywood':'Plywood',
        'PreCast':'PreCast',
        'Stone':'Stone',
        'Stucco':'Stucco',
        'VinylSd':'Vinyl Siding',
        'Wd Sdng':'Wood Siding',
        'Wd Shng':'Wood Shingles'
        },
    'type':'num_desc'
    }
mappings['Fence'] = {
    'values':{
        'GdPrv':'Good Privacy',
        'MnPrv':'Minimum Privacy',
        'GdWo':'Good Wood',
        'MnWw':'Minimum Wood/Wire',
        'NA':'No Fence'
        },
    'type':'num_desc'
    }
mappings['FireplaceQu'] = {
    'values':{
        'Ex':'Excellent - Exceptional Masonry Fireplace',
        'Gd':'Good - Masonry Fireplace in main level',
        'TA':'Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement',
        'Fa':'Fair - Prefabricated Fireplace in basement',
        'Po':'Poor - Ben Franklin Stove',
        'NA':'No Fireplace'
        },
    'type':'num_desc'
    }
mappings['Foundation'] = {
    'values':{
        'BrkTil':'Brick & Tile',
        'CBlock':'Cinder Block',
        'PConc':'Poured Contrete',
        'Slab':'Slab',
        'Stone':'Stone',
        'Wood':'Wood'
        },
    'type':'num_desc'
    }
mappings['Functional'] = {
    'values':{
        'Typ':'Typical Functionality',
        'Min1':'Minor Deductions 1',
        'Min2':'Minor Deductions 2',
        'Mod':'Moderate Deductions',
        'Maj1':'Major Deductions 1',
        'Maj2':'Major Deductions 2',
        'Sev':'Severely Damaged',
        'Sal':'Salvage only'
        },
    'type':'num_desc'
    }
mappings['GarageCond'] = {
    'values':{
        'Ex':'Excellent',
        'Gd':'Good',
        'TA':'Typical/Average',
        'Fa':'Fair',
        'Po':'Poor',
        'NA':'No Garage'
        },
    'type':'num_desc'
    }
mappings['GarageFinish'] = {
    'values':{
        'Fin':'Finished',
        'RFn':'Rough Finished',
        'Unf':'Unfinished',
        'NA':'No Garage'
        },
    'type':'num_desc'
    }
mappings['GarageQual'] = {
    'values':{
        'Ex':'Excellent',
        'Gd':'Good',
        'TA':'Typical/Average',
        'Fa':'Fair',
        'Po':'Poor',
        'NA':'No Garage'
        },
    'type':'num_desc'
    }
mappings['GarageType'] = {
    'values':{
        '2Types':'More than one type of garage',
        'BuiltIn':'Built-In (Garage part of house - typically has room above garage)',
        'Basment':'Basement Garage',
        'Attchd':'Attached to home',
        'Detchd':'Detached from home',
        'CarPort':'Car Port',
        'NA':'No Garage'
        },
    'type':'num_desc'
    }
mappings['Heating'] = {
    'values':{
        'Floor':'Floor Furnace',
        'GasA':'Gas forced warm air furnace',
        'GasW':'Gas hot water or steam heat',
        'Grav':'Gravity furnace',
        'OthW':'Hot water or steam heat other than gas',
        'Wall':'Wall furnace'
        },
    'type':'num_desc'
    }
mappings['HeatingQC'] = {
    'values':{
        'Ex':'Excellent',
        'Gd':'Good',
        'TA':'Typical/Average',
        'Fa':'Fair',
        'Po':'Poor'
        },
    'type':'num_desc'
    }
mappings['HouseStyle'] = {
    'values':{
        '2.5Fin':'Two and one-half story: 2nd level finished',
        '2.5Unf':'Two and one-half story: 2nd level unfinished',
        '2Story':'Two story',
        '1.5Fin':'One and one-half story: 2nd level finished',
        '1.5Unf':'One and one-half story: 2nd level unfinished',
        'SFoyer':'Split Foyer',
        'SLvl':'Split Level',
        '1Story':'One story'
        },
    'type':'num_desc'
    }
mappings['KitchenQual'] = {
    'values':{
        'Ex':'Excellent',
        'Gd':'Good',
        'TA':'Typical/Average',
        'Fa':'Fair',
        'Po':'Poor'
        },
    'type':'num_desc'
    }
mappings['LandContour'] = {
    'values':{
        'Lvl':'Near Flat/Level',
        'Bnk':'Banked - Quick and significant rise from street grade to building',
        'HLS':'Hillside - Significant slope from side to side',
        'Low':'Depression'
        },
    'type':'num_desc'
    }
mappings['LandSlope'] = {
    'values':{
        'Gtl':'Gentle slope',
        'Mod':'Moderate Slope',
        'Sev':'Severe Slope'
        },
    'type':'num_desc'
    }
mappings['LotConfig'] = {
    'values':{
        'Inside':'Inside lot',
        'CulDSac':'Cul-de-sac',
        'Corner':'Corner lot',
        'FR2':'Frontage on 2 sides of property',
        'FR3':'Frontage on 3 sides of property'
        },
    'type':'num_desc'
    }
mappings['LotShape'] = {
    'values':{
        'Reg':'Regular',
        'IR1':'Slightly irregular',
        'IR2':'Moderately Irregular',
        'IR3':'Irregular'
        },
    'type':'num_desc'
    }
    
mappings['MSSubClass'] = {
    'values':{
        '20':'1-STORY 1946 & NEWER ALL STYLES',
        '30':'1-STORY 1945 & OLDER',
        '40':'1-STORY W/FINISHED ATTIC ALL AGES',
        '45':'1-1/2 STORY - UNFINISHED ALL AGES',
        '50':'1-1/2 STORY FINISHED ALL AGES',
        '60':'2-STORY 1946 & NEWER',
        '70':'2-STORY 1945 & OLDER',
        '75':'2-1/2 STORY ALL AGES',
        '80':'SPLIT OR MULTI-LEVEL',
        '85':'SPLIT FOYER',
        '90':'DUPLEX - ALL STYLES AND AGES',
        '120':'1-STORY PUD (Planned Unit Development) - 1946 & NEWER',
        '150':'1-1/2 STORY PUD - ALL AGES',
        '160':'2-STORY PUD - 1946 & NEWER',
        '180':'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER',
        '190':'2 FAMILY CONVERSION - ALL STYLES AND AGES'
        },
    'type':'num_desc'
    }

mappings['MSZoning'] = {
    'values':{
        'RL':'Residential Low density',
        'RM':'Residential Medium Density',
        'C (all)':'Commercial',
        'FV':'Floating Village Residential',
        'RH':'Residential High Density',
        'NA':'Dont know'
        },
    'type':'num_desc'
    }
    
mappings['MasVnrType'] = {
    'values':{
        'Stone':'Stone',
        'BrkCmn':'Brick Common',
        'BrkFace':'Brick Face',
        'CBlock':'Cinder Block',
        'None':'None'
        },
    'type':'num_desc'
    }
mappings['MiscFeature'] = {
    'values':{
        'Elev':'Elevator',
        'Gar2':'2nd Garage (if not described in garage section)',
        'Shed':'Shed (over 100 SF)',
        'TenC':'Tennis Court',
        'Othr':'Other',
        'NA':'None'
        },
    'type':'num_desc'
    }
mappings['Neighborhood'] = {
    'values':{
        'Blmngtn':'Bloomington Heights',
        'Blueste':'Bluestem',
        'BrDale':'Briardale',
        'BrkSide':'Brookside',
        'ClearCr':'Clear Creek',
        'CollgCr':'College Creek',
        'Crawfor':'Crawford',
        'Edwards':'Edwards',
        'Gilbert':'Gilbert',
        'IDOTRR':'Iowa DOT and Rail Road',
        'MeadowV':'Meadow Village',
        'Mitchel':'Mitchell',
        'NAmes':'North Ames',
        'NoRidge':'Northridge',
        'NPkVill':'Northpark Villa',
        'NridgHt':'Northridge Heights',
        'NWAmes':'Northwest Ames',
        'OldTown':'Old Town',
        'SWISU':'South & West of Iowa State University',
        'Sawyer':'Sawyer',
        'SawyerW':'Sawyer West',
        'Somerst':'Somerset',
        'StoneBr':'Stone Brook',
        'Timber':'Timberland',
        'Veenker':'Veenker'
        },
    'type':'onehot'
    }
    
mappings['PavedDrive'] = {
    'values':{
        'Y':'Paved',
        'P':'Partial Pavement',
        'N':'Dirt/Gravel'
        },
    'type':'num_desc'
    }
mappings['PoolQC'] = {
    'values':{
        'Ex':'Excellent',
        'Gd':'Good',
        'Fa':'Fair',
        'NA':'No Pool'
        },
    'type':'num_desc'
    }
mappings['RoofMatl'] = {
    'values':{
        'ClyTile':'Clay or Tile',
        'CompShg':'Standard (Composite) Shingle',
        'Membran':'Membrane',
        'Metal':'Metal',
        'Roll':'Roll',
        'Tar&Grv':'Gravel & Tar',
        'WdShake':'Wood Shakes',
        'WdShngl':'Wood Shingles'
        },
    'type':'num_desc'
    }
mappings['RoofStyle'] = {
    'values':{
        'Flat':'Flat',
        'Gable':'Gable',
        'Gambrel':'Gabrel (Barn)',
        'Hip':'Hip',
        'Mansard':'Mansard',
        'Shed':'Shed'
        },
    'type':'num_desc'
    }
    
mappings['SaleCondition'] = {
    'values':{
        'Normal':'Normal Sale',
        'Abnorml':'Abnormal Sale -  trade, foreclosure, short sale',
        'AdjLand':'Adjoining Land Purchase',
        'Alloca':'Allocation - two linked properties with separate deeds, typically condo with a garage unit',
        'Family':'Sale between family members',
        'Partial':'Home was not completed when last assessed (associated with New Homes)'
        },
    'type':'num_desc'
    }
mappings['SaleType'] = {
    'values':{
        'WD':'Warranty Deed - Conventional',
        'CWD':'Warranty Deed - Cash',
        'VWD':'Warranty Deed - VA Loan',
        'New':'Home just constructed and sold',
        'COD':'Court Officer Deed/Estate',
        'Con':'Contract 15% Down payment regular terms',
        'ConLw':'Contract Low Down payment and low interest',
        'ConLI':'Contract Low Interest',
        'ConLD':'Contract Low Down',
        'Oth':'Other'
        },
    'type':'num_desc'
    }
mappings['Street'] = {
    'values':{
        'Pave':'Paved',
        'Grvl':'Gravel'
        },
    'type':'num_desc'
    }
mappings['Utilities'] = {
    'values':{
        'AllPub':'All public Utilities (E,G,W,& S)',
        'NoSeWa':'Electricity and Gas Only'
        },
    'type':'num_desc'
    }
nan_NA = ['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature','MSZoning']
medians = ['MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea',]
#zeros = ['LotFrontage','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea',]
def clean_lot_frontage(df):
    for i, row in df.iterrows():
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

def clean_data(df, exclude=[], mappings={}, medians=[],all_num=False):
    df = clean_lot_frontage(df)
    df = clean_nan_saletype(df)
    df = clean_nan_masvnrtype(df)
    df = clean_nan_electrical(df)
    df = clean_nan_exterior(df)
    df = clean_nan_kitchen_quality(df)
    df = clean_nan_functional(df)
    df = clean_nan_utilities(df)
    columns = df.columns
    for col in nan_NA:
        df[col] = df[col].fillna('NA')
    for col in columns:
        col_type = df[col].dtype
        if col_type == 'object' and col in mappings:
            #p(col+' unique',unique)
            if mappings[col]['type'] == 'num_desc' or all_num == True:
                map_array = list(mappings[col]['values'].keys())
                to_map = dict(zip(map_array[::-1],range(len(map_array))))
                df[col] = df[col].map(to_map)
            elif mappings[col]['type'] == 'onehot':
                df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
                df = df.drop(col, axis=1)
            else:
                print('unrecognized type!!!!!!!')
    for med in medians:
        df[med] = df[med].map(lambda x: np.nan if x == 'NA' else x)
        df[med] = df[med].fillna(df[med].median())
        #p(col+' type',col_type)
    #df = df.fillna('NA')
    return df

def isolation_tree(Axis_1,Axis_2,label=None,color=None):
    data = pd.concat([Axis_1,Axis_2],axis=1).copy()
#    print('data',type(data),"\n",data)
#    print('Axis_1 uniques',type(Axis_1),"\n",Axis_1.unique())
#    print('Axis_2 uniques',type(Axis_2),"\n",Axis_2.unique())
    clf = IsolationForest(max_samples=100, random_state=0)   
    clf.fit(data)
    xx, yy = np.meshgrid(np.linspace(Axis_1.min(), Axis_1.max(), len(Axis_1.unique())), np.linspace(Axis_2.min(), Axis_2.max(), len(Axis_2.unique())))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    a = plt.scatter(Axis_1, Axis_2, c='white', s=20, edgecolor='k')
    plt.axis('tight')
    plt.xlim((Axis_1.min(), Axis_1.max()))
    plt.ylim((Axis_2.min(), Axis_2.max()))
    #print('Axis_1',type(Axis_1),Axis_1.shape,"\n",Axis_1,"\nname",Axis_1.name)
    #print('Axis_2',type(Axis_2),Axis_2.shape,"\n",Axis_2,"\nname",Axis_2.name)
    plt.legend([a],
               [Axis_1.name+' / '+Axis_2.name],
               loc="upper left")
    plt.show()
    
def pair_grid_visu(data, predict_column):
    #g = sns.pairplot(visu, hue="Survived", palette="husl")
    #g = sns.pairplot(visu, hue="Sex", palette="husl")
    g1 = sns.PairGrid(data, hue=predict_column, palette="husl")
    g1 = g1.map_diag(plt.hist)
    g1 = g1.map_upper(plt.scatter)
    g1 = g1.map_lower(sns.swarmplot)
    g1 = g1.add_legend()
    
    g = sns.PairGrid(data, palette="husl")
    g = g.map_diag(plt.hist)
    #g = g.map_offdiag(plt.scatter) 
    #g = g.map_upper(plt.scatter)
    g = g.map_upper(isolation_tree)
    g = g.map_lower(sns.swarmplot)
    g = g.add_legend()    
    


select = ['OverallQual','GrLivArea','TotalBsmtSF']
#select = ['GrLivArea','OverallQual']
drops = ['Id','SalePrice']


train = clean_data(train, mappings=mappings, medians=medians)#, all_num=True)
test = clean_data(test, mappings=mappings, medians=medians)
#sub = clean_data(sub, mappings=mappings, medians=medians)

#retrait d'outliers
train = train[train['GrLivArea'] < 4600]
train = train[train['GarageArea'] < 1200]
train = train[train['TotalBsmtSF'] < 6000]
train = train[train['1stFlrSF'] < 4000]

#isof = IsolationForest(contamination=0.05, random_state=0)   
#mask_isof = isof.fit(train[select]).predict(train[select])
#train_filtered = train[mask_isof==1]
#train = train_filtered

correlations = get_correlations(train, predict_column, 'correlations après clean,')

nb_selected_features = 78
#select_visu = list(correlations.keys()[0:nb_selected_features+1])
#visu = train[select_visu].copy()
#isolation_tree(visu['SalePrice'],visu['GrLivArea'])
#pair_grid_visu(visu, predict_column)   


print('empty columns : ',na_rows(train))

#for var in select:
#    data = pd.concat([train[predict_column], train[var]], axis=1)
#    data.plot.scatter(x=var, y=predict_column, ylim=(0,800000));

#création des tableaux d'entrainement et de cible, des tableaux de validation intermédiaire et le tableau à prédire au final
select = list(correlations.keys()[1:nb_selected_features+1])
#print('selected columns : ',select)
#X = train.drop(drops, axis=1).iloc[:,1:].copy()
#X = train[select].copy()

#X = train.drop(drops, axis=1).copy()
y = train[predict_column].copy()
#X_kaggle = test.drop('Id', axis=1).copy()
#all_data = X.append(X_kaggle)


X = train[select].copy()
X_kaggle = test[select].copy()

print('X.shape',X.shape)
print('X_kaggle.shape',X_kaggle.shape)

#%%

scaler = MinMaxScaler()
scaler.fit(X)


rng = np.random.randint(100)
names = [
        #'LinearRegression',
        'Ridge',
        'RidgeCV',
        'Lasso',
        #'MultiTaskLasso',
        #'ElasticNet',
        #'ElasticNetCV',
        #'MultiTaskElasticNet',
        #'Lars',
        'LassoLars',
        'OrthogonalMatchingPursuit',
        'BayesianRidge',
        #'ARDRegression',
        'SGDRegressor',
        #'PassiveAggressiveRegressor',
        'HuberRegressor',
        #'MSIASolver',
        'RandomForestRegressor',
        'GradientBoostingRegressor'
        ]
classifier = [
        #LinearRegression(),
        Ridge(),
        RidgeCV(),
        Lasso(),
        #MultiTaskLasso(),
        #ElasticNet(),
        #ElasticNetCV(),
        #MultiTaskElasticNet(),
        #Lars(),
        LassoLars(),
        OrthogonalMatchingPursuit(),
        BayesianRidge(),
        #ARDRegression(),
        SGDRegressor(),
        #PassiveAggressiveRegressor(),
        HuberRegressor(),
        #MSIASolver.MSIASolver(),
        RandomForestRegressor(),
        GradientBoostingRegressor()
        ]
#names = ['MSIASolver']
#classifier = [MSIASolver.MSIASolver()]
names = ['GradientBoostingRegressor']
classifier = [GradientBoostingRegressor()]

#fonction de cross_validation
best_clf = None
best_clf_name = None
best_score = 0
idx = 0


clf_results = pd.DataFrame(index=range(len(classifier)), columns=['name','score'])
def run_kfold(X, y, clf, name, best_clf, best_score, best_clf_name, clf_results, scaler):
    kf = KFold(len(X), n_folds=15)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        #accuracy = explained_variance_score(y_test, predictions)
        accuracy = clf.score(X_test, y_test)
        outcomes.append(accuracy)
        print(name, " - Fold {0} accuracy: {1}".format(fold, accuracy),' RMSE : ',mean_squared_error(y_test, predictions))
    mean_outcome = np.mean(outcomes)
    std_outcome = np.std(outcomes)
    print("\n",name," - Mean Accuracy: {0}".format(mean_outcome),' +/- ',std_outcome)
    print("\n"+'---------------------------------------------------------'+"\n")
    if mean_outcome > best_score:
        best_score = mean_outcome
        best_clf = clf
        best_clf_name = name
    clf_results.at[idx,'name'] = name
    clf_results.at[idx,'score'] = mean_outcome
    return (best_clf, best_score, best_clf_name)

#entrainement des différents classifieurs
for name, clf in zip(names,classifier):
    best_clf, best_score, best_clf_name = run_kfold(X, y, clf, name, best_clf, best_score, best_clf_name, clf_results, scaler)
    idx+=1

print('Résultats des classifieurs'+"\n")
clf_results.sort_values(by='score', ascending=False, inplace=True)
print(clf_results,"\n")
if len(clf_results)>1:
    plt.figure()
    sns.barplot(x='score',y='name',data=clf_results,palette="Set1")
    plt.show()
print("\n"+'Best classifier = ',best_clf_name)
print('with score = ',best_score,"\n")

#préparation des différents paramètres du classifieur à tester

clf_params = {}
clf_params['LinearRegression'] = {
              'fit_intercept': [False, True],
              'normalize': [False, True],
              'n_jobs': [1, 2, 3, 4]
             }
clf_params['Ridge'] = {
              'alpha': [1, 3, 6, 10],
              'fit_intercept': [False, True],
              'normalize': [False, True]
             }
clf_params['Lasso'] = {
              'alpha': [1, 3, 6, 10],
              'fit_intercept': [False, True],
              'normalize': [False, True],
              'precompute': [False, True]
             }
clf_params['Lars'] = {
              'fit_intercept': [False, True],
              'verbose': [1, 3, 6, 10],
              'normalize': [False, True],
              'precompute': [False, True]
             }
clf_params['LassoLars'] = {
              'alpha': [1, 3, 6, 10],
              'fit_intercept': [False, True],
              'verbose': [1, 3, 6, 10],
              'normalize': [False, True],
              'precompute': [False, True]
             }
clf_params['OrthogonalMatchingPursuit'] = {
              'n_nonzero_coefs': [1, 3, 6, 10],
              'fit_intercept': [False, True],
              'normalize': [False, True],
              'precompute': [False, True]
             }
clf_params['BayesianRidge'] = {
              'alpha': [0.0000001, 0.00001, 0.001, 0.1],
              'fit_intercept': [False, True],
              'normalize': [False, True],
              'precompute': [False, True]
             }
clf_params['SGDRegressor'] = {
              'alpha': [0.0000001, 0.00001, 0.001, 0.1],
              'penalty': ['none', 'l2', 'l1', 'elasticnet'],
              'fit_intercept': [False, True]
             }
clf_params['HuberRegressor'] = {
              'alpha': [0.0000001, 0.00001, 0.001, 0.1],
              'epsilon': [1, 1.35, 2, 5],
              'fit_intercept': [False, True]
             }
clf_params['HuberRegressor'] = {
              'alpha': [0.0000001, 0.00001, 0.001, 0.1],
              'epsilon': [1, 1.35, 2, 5],
              'fit_intercept': [False, True]
             }
clf_params['RandomForestRegressor'] = {
        'n_estimators': [60, 100, 150],
              'max_features': ['log2', 'sqrt','auto'],
              'criterion': ['mse', 'mae'],
              #'max_depth': [None, 8, 32, 64],
              #'min_samples_split': [0.1, 0.2, 0.5, 0.7, 1.0],
              #'min_samples_leaf': [1,2,5]
             }
clf_params['GradientBoostingRegressor'] = {
        'n_estimators': [60, 100, 150],
              'loss': ['ls', 'lad','huber','quantile'],
              #'criterion': ['mse', 'mae'],
              #'max_depth': [None, 8, 32, 64],
              #'min_samples_split': [0.1, 0.2, 0.5, 0.7, 1.0],
              #'min_samples_leaf': [1,2,5]
             }

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(explained_variance_score)
# Run the grid search
grid_obj = GridSearchCV(best_clf, clf_params[best_clf_name], scoring=acc_scorer)


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train_columns = X_kaggle.columns
X_kaggle = scaler.transform(X_kaggle)  

grid_obj = grid_obj.fit(X_train, y_train)
# Set the clf to the best combination of parameters
best_clf = grid_obj.best_estimator_
# Fit the best algorithm to the data.
best_clf.fit(X_train, y_train)
#création des prédictions pour la validation intermédiaire
predictions = best_clf.predict(X_test)
print("\n",'Accuracy on test set = ', explained_variance_score(y_test, predictions))

#predicted_thetas = best_clf.get_predicted_thetas()
#print("\n",'predicted thetas on test set = ', predicted_thetas)
#best_clf.coef_ = predicted_thetas[1:].reshape(1,predicted_thetas.shape[0]-1)
#print("\n",'best_clf.coef_ ', best_clf.coef_)


#X_kaggle.describe()
#X_kaggle.info()
#cnt = 0
#max_ = len(X_kaggle.columns)
#while cnt<max_:
#    cnt_init = cnt
#    cnt+=15
#    if cnt>max_:
#        cnt=max_
#    print(X_kaggle.iloc[:,cnt_init:cnt].info()) 
 

print('X.shape',X.shape)
print('X_kaggle.shape',X_kaggle.shape)

#X_kaggle['Utilities_NoSeWa'] = 0
#création des prédictions sur l'échantillon à tester
predictions = best_clf.predict(X_kaggle)

#création du tableau de sortie pour Kaggle
output = pd.DataFrame({ 'Id' : Id_test, predict_column: predictions })
output = output.set_index('Id')
output.to_csv('Housing_Prices_Kaggle_Sub.csv')

#hack pour l'affichage
test['Id'] = Id_test
test[predict_column] = predictions

if hasattr(best_clf, 'feature_importances_') and hasattr(best_clf, 'estimators_'):
    importances = best_clf.feature_importances_
    #print('importances',"\n",importances)
    #std = np.std([tree.feature_importances_ for tree in best_clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("\n","Feature ranking:")
    for f in range(X_train.shape[1]):
        print("%d. %s (%f)" % (f + 1, X_train_columns[indices[f]], importances[indices[f]]))
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    #plt.bar(range(X_train.shape[1]), importances[indices], color="b", yerr=std[indices], align="center")
    plt.bar(range(X_train.shape[1]), importances[indices], color="g", align="center")
    plt.xticks(range(X_train.shape[1]), X_train_columns[indices], rotation=45)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()
    
if hasattr(best_clf, 'coef_'):
    importances = best_clf.coef_
    print('importances',type(importances),importances.shape,importances)
    indices = np.argsort(importances)[::-1]
    for f in range(len(X_train_columns)):
#        print('f',f)
#        p('indices',indices)
#        print('indices[0,f]',indices[f])
#        p('X_train_columns',X_train_columns)
#        p('importances',importances)
        x_train_col = X_train_columns[indices[f]]
        importance = importances[indices[f]]
        print("%d. %s (%f)" % (f + 1, x_train_col, importance))
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances (relative)")
    #plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.bar(range(X_train.shape[1]), importances[indices[0]], color="b", align="center")
    plt.xticks(range(X_train.shape[1]), X_train_columns[indices[0]], rotation=45)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()
    
    # Plot the feature importances of the forest
    df_abs = pd.DataFrame(importances).abs()
    indices = np.argsort(np.array([list(df_abs[0])]))
    df_abs_sorted = df_abs.sort_values(by=0,ascending= False)
    importances = np.array([list(df_abs_sorted[0])])
    plt.figure()
    plt.title("Feature importances (absolute)")
    #plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.bar(range(X_train.shape[1]), importances, color="b", align="center")
    plt.xticks(range(X_train.shape[1]), X_train_columns[indices[::-1]], rotation=45)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()
    
#    
#def plot_confusion_matrix(cm, classes,
#                          normalize=False,
#                          title='Confusion matrix',
#                          cmap=plt.cm.Blues):
#    """
#    This function prints and plots the confusion matrix.
#    Normalization can be applied by setting `normalize=True`.
#    """
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
#
#    print(cm)
#
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
#    plt.yticks(tick_marks, classes)
#
#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")
#
#    plt.tight_layout()
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')
#
## Compute confusion matrix
#cnf_matrix = confusion_matrix(y_test, predictions)
#np.set_printoptions(precision=2)
#
## Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names,
#                      title='Confusion matrix, without normalization')
#
## Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                      title='Normalized confusion matrix')
#
#plt.show()