import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cross_validation import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, explained_variance_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

#fonction utile pour le tracing
def p(mess,obj):
    """Useful function for tracing"""
    if hasattr(obj,'shape'):
        print(mess,type(obj),obj.shape,"\n",obj)
    else:
        print(mess,type(obj),"\n",obj)

class MSIA_SK_Solver():
    """Solver class
    """
    def __init__(self):
        self.X = None
        self.Y = None
        self.n_dimensions = None
        self.n_samples = None
        self.range_x = None
        self.predict_column = None
        self.problem_type = None
        self.estimators = None
    
    def info_all(self, df=None):
        if df is None:
            df = self.X
        cnt = 0
        max_ = len(df.columns)
        while cnt<max_:
            cnt_init = cnt
            cnt+=15
            if cnt>max_:
                cnt=max_
            print(df.iloc[:,cnt_init:cnt].info())

    def na_rows(self, df=None):
        if df is None:
            df = self.X
        return df.columns[df.isna().any()] 
    
    def null_report(self, df=None):
        if df is None:
            df = self.X
        total = df.isnull().sum().sort_values(ascending=False)
        percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        #print('Missing Data\n',missing_data.head(20))
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Null values")
        #plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
        array = missing_data[missing_data['Total']>0]
        n_features = len(array)
        print('n_features',n_features)
        print('array',type(array),array.shape,'\n',array)
        print('array.index',array.index)
        plt.bar(range(n_features), array['Total'], color="b", align="center")
        plt.xticks(np.arange(-0.3,n_features-0.3,1), array.index, rotation=45)
        plt.xlim([-1, n_features])
        plt.xlabel("Features")
        plt.ylabel("Number of null values")
        plt.show()
    
    def get_correlations(self, df=None, predict_column=None, mess='correlations'):
        if df is None:
            df = self.X
        if predict_column is None:
            predict_column = self.predict_column
        #calcul et affichage des facteurs de corrélation des variables
        #methods = ['pearson', 'kendall', 'spearman']
        correlation = df.corr(method='pearson')
        #print('correlation',correlation)
        correlations_relatives = correlation[predict_column].sort_values(ascending= False)
        print("\n"+mess+' relatives = '+"\n",correlations_relatives,"\n")
        
        n_features = len(correlations_relatives)
        
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances (relative)")
        #plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
        plt.bar(range(n_features), correlations_relatives, color="b", align="center")
        plt.xticks(range(n_features), correlations_relatives.keys(), rotation=45)
        plt.xlim([-1, n_features])
        plt.show()
        
        correlations_absolues = correlation[predict_column].abs().sort_values(ascending= False)
        print(mess+' absolues = ',type(correlations_absolues),"\n",correlations_absolues,"\n","\n")
        
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances (absolute)")
        #plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
        plt.bar(range(n_features), correlations_absolues, color="b", align="center")
        plt.xticks(range(n_features), correlations_absolues.keys(), rotation=45)
        plt.xlim([-1, n_features])
        plt.show()
        
        #return correlations_absolues
    
    def heatmap(self, df=None, predict_column=None, nb = 15):
        if df is None:
            df = self.X
        corrmat = df.corr()
        columns = corrmat.nlargest(nb, predict_column)[predict_column].index
        corrmat = corrmat.nlargest(nb, predict_column)[columns]
        #f, ax = plt.subplots(figsize=(12, 9))
        plt.figure()
        sns.heatmap(corrmat, vmax=.8, square=True)
        plt.show()
    
    def clustermap(self, df=None, predict_column=None, nb = 20):
        if df is None:
            df = self.X
        corrmat = df.corr()
        columns = corrmat.nlargest(nb, predict_column)[predict_column].index
        corrmat = corrmat.nlargest(nb, predict_column)[columns]
        #f, ax = plt.subplots(figsize=(12, 9))
        plt.figure()
        sns.clustermap(corrmat, vmax=.8, square=True)
        plt.show()
        
    def corr_mat(self, df=None, predict_column=None, nb = 20):
        if df is None:
            df = self.X
        if predict_column is None:
            predict_column = self.predict_column
        corrmat = df.corr()
        columns = corrmat.nlargest(nb, predict_column)[predict_column].index
        corrmat = corrmat.nlargest(nb, predict_column)[columns]
        plt.figure()
        cols = corrmat.nlargest(nb, predict_column)[predict_column].index
        cm = np.corrcoef(df[cols].values.T)
        sns.set(font_scale=2.0)
        sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols.values, xticklabels=cols.values)
        plt.show()

    def pca_analysis(self, df=None, n_comp = 20):
        if df is None:
            df = self.X
        print('\nRunning PCA ...')
        pca = PCA(n_components=n_comp, svd_solver='full', random_state=1001)
        df_pca = pca.fit_transform(df)
        print('df_pca',df_pca)
        print('Explained variance: %.4f' % pca.explained_variance_ratio_.sum())
        
        print('Individual variance contributions:')
        print('noise:',pca.noise_variance_)
        for j in range(pca.n_components_):
            print('ratio',pca.explained_variance_ratio_[j],'var',pca.explained_variance_[j],'val',pca.singular_values_[j],'mean',pca.mean_[j])
        print(np.mean(pca.get_covariance(),axis=1))
        print('columns',df.columns)
        
    def isolation_tree(self, Axis_1,Axis_2,label=None,color=None):
        data = pd.concat([Axis_1,Axis_2],axis=1).copy()
    #    print('data',type(data),"\n",data)
    #    print('Axis_1 uniques',type(Axis_1),"\n",Axis_1.unique())
    #    print('Axis_2 uniques',type(Axis_2),"\n",Axis_2.unique())
        estimator = IsolationForest(max_samples=100, random_state=0)   
        estimator.fit(data)
        xx, yy = np.meshgrid(np.linspace(Axis_1.min(), Axis_1.max(), len(Axis_1.unique())), np.linspace(Axis_2.min(), Axis_2.max(), len(Axis_2.unique())))
        Z = estimator.decision_function(np.c_[xx.ravel(), yy.ravel()])
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
        
    def pair_grid_visu(self, df=None, predict_column=None):
        if df is None:
            df = self.X
        if predict_column is None:
            predict_column = self.predict_column
        #g = sns.pairplot(visu, hue="Survived", palette="husl")
        #g = sns.pairplot(visu, hue="Sex", palette="husl")
        g1 = sns.PairGrid(df, hue=predict_column, palette="husl")
        g1 = g1.map_diag(plt.hist)
        g1 = g1.map_upper(plt.scatter)
        g1 = g1.map_lower(sns.swarmplot)
        g1 = g1.add_legend()
        
        g = sns.PairGrid(df, palette="husl")
        g = g.map_diag(plt.hist)
        #g = g.map_offdiag(plt.scatter) 
        #g = g.map_upper(plt.scatter)
        g = g.map_upper(self.isolation_tree)
        g = g.map_lower(sns.swarmplot)
        g = g.add_legend()    
        
    def plot_scatter(self, df=None, axis_1= None, axis_2=None):
        if df is None:
            df = self.X
        if axis_1 is None:
            axis_1 = self.X.columns[0]
        if axis_2 is None:
            axis_2 = self.predict_column
        plt.figure()
        plt.scatter(x=df[axis_1], y=df[axis_2])
        plt.xlabel(axis_1)
        plt.ylabel(axis_2)
        plt.show()
        
    def plot_most_important(self, df=None, axis=None, corr=None, nb=18, start=1):
        if df is None:
            df = self.X
        if axis is None:
            axis = self.predict_column
        if corr is None:
            corr = df.corr()  
            corr = corr[axis].abs().sort_values(ascending= False)
        for i in range(start, start+nb):
            self.plot_scatter(df, corr.keys()[i], axis)
        
    def first_analysis(self, df=None, predict_column=None):
        if df is None:
            df = self.X
        if predict_column is None:
            predict_column = self.predict_column
        else:
            self.predict_column = predict_column
        self.get_correlations(df, predict_column)
        self.heatmap(df, predict_column)
        self.clustermap(df, predict_column)
        self.corr_mat(df, predict_column)
        #self.plot_most_important(df)
        
    def feed_data(self, df, predict_column, n_first=None, id_col=0):
        #self.Y = np.array(df[predict_column].copy().tolist()).reshape((df.shape[0],1))
        self.id_col = id_col
        if id_col == 0:
            self.id_col = df.columns[0]
        else:
            self.id_col = id_col
        self.Y = df[predict_column].copy()
        self.X = df.drop(predict_column, axis=1).copy()
        self.n_dimensions = self.X.shape[1]
        self.n_samples = df.shape[0]
        self.columns = df.columns
        self.range_x = np.max(np.abs(self.X))
        self.n_first = n_first
        if self.n_first is not None:
            self.n_dimensions = self.n_first
            corrmat = df.corr()
            columns = corrmat.nlargest(self.n_first, predict_column)[predict_column].index
            corrmat = corrmat.nlargest(self.n_first, predict_column)[columns]
            correlations = corrmat[predict_column].abs().sort_values(ascending= False)
            self.select = list(correlations.keys()[1:self.n_first+1])
            self.X = self.X[self.select]
#        print('auto_clean:')
#        print('n_dimensions:',self.n_dimensions)
#        print('n_samples:',self.n_samples)
#        print('range_x:',self.range_x)
#        print('columns:',self.columns)
        return self
    
    def auto_clean(self, df=None, predict_column=None):
        if df is None:
            df = self.X
        else:
            self.X = df.copy()
        if predict_column is None:
            predict_column = self.predict_column
        else:
            self.predict_column = predict_column
        mapping = {}
        for col in df.columns:
            col_type = df[col].dtype
            if col_type == 'object':
                unique = df[col].unique()
                means = []
                for val in unique:
                    mean = np.mean(df[df[col]==val][predict_column])
                    means.append(mean)
                indices = np.argsort(means)
                indices2 = np.zeros(len(indices))
                for i, j in zip(indices,range(len(indices))):
                    indices2[i] = j
                map_array = dict(zip(unique,indices2))
                mapping[col] = map_array
                print('col',col)
#                print('unique',unique)
#                print('means',means)
#                print('indices',indices)
#                print('indices2',indices2)
                print('map_array',map_array)
                df[col] = df[col].map(map_array)
        self.feed_data(df, predict_column)
        return df, mapping
    
    def map_clean(self, df, mapping):
        for col in df.columns:
            col_type = df[col].dtype
            if col_type == 'object' and col in mapping:
                map_array = mapping[col]
                df[col] = df[col].map(map_array)
        return df
    
    def map_clean2(self, df, mapping):
        for col in df.columns:
            col_type = df[col].dtype
            if col_type == 'object' and col in mapping:
                if mapping[col]['type'] == 'num':
                    map_array = list(mapping[col]['values'].keys())
                    to_map = dict(zip(map_array[::-1],range(len(map_array))))
                    df[col] = df[col].map(to_map)
                elif mapping[col]['type'] == 'onehot':
                    df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
                    df = df.drop(col, axis=1)
                else:
                    print('unrecognized type!!!!!!!')
        return df
        
    def detect_problem_type(self):
        """Solver: detect problem type
        """
        if self.problem_type == None:
            #if(self.Y.shape[1]==1):
            self.problem_type = 'linear'
            min_ = self.Y.min(axis=0)
            if (self.Y.dtype == 'int32' or self.Y.dtype == 'int64' or self.Y.dtype == 'bool') and min_ >= 0:
                unique = np.unique(self.Y.astype(float))
                if len(unique) < 10:
                    test = True
                    for item in unique:
                        if item.is_integer() == False:
                            test = False
                    if test == True:
                        self.problem_type = 'classification'
            if self.problem_type == 'classification':
                self.set_logistic_regressors()
            else:
                self.set_linear_regressors()

    def set_linear_regressors(self):
        self.estimators = [
            #LinearRegression(),
            Ridge(),
            RidgeCV(),
            Lasso(),
            #MultiTaskLasso(),
            ElasticNet(),
            ElasticNetCV(),
            #MultiTaskElasticNet(),
            Lars(),
            LassoLars(),
            OrthogonalMatchingPursuit(),
            BayesianRidge(),
            #ARDRegression(),
            SGDRegressor(),
            PassiveAggressiveRegressor(),
            HuberRegressor(),
            RandomForestRegressor(),
            GradientBoostingRegressor()
        ]
        self.estimator_params = {}
        self.estimator_params['LinearRegression'] = {
                      'fit_intercept': [False, True],
                      'normalize': [False, True],
                      'n_jobs': [1, 2, 3, 4]
                     }
        self.estimator_params['Ridge'] = {
                      'alpha': [1, 3, 6, 10],
                      'fit_intercept': [False, True],
                      'normalize': [False, True]
                     }
        self.estimator_params['Lasso'] = {
                      'alpha': [1, 3, 6, 10],
                      'fit_intercept': [False, True],
                      'normalize': [False, True],
                      'precompute': [False, True]
                     }
        self.estimator_params['Lars'] = {
                      'fit_intercept': [False, True],
                      'verbose': [1, 3, 6, 10],
                      'normalize': [False, True],
                      'precompute': [False, True]
                     }
        self.estimator_params['LassoLars'] = {
                      'alpha': [1, 3, 6, 10],
                      'fit_intercept': [False, True],
                      'verbose': [1, 3, 6, 10],
                      'normalize': [False, True],
                      'precompute': [False, True]
                     }
        self.estimator_params['OrthogonalMatchingPursuit'] = {
                      'n_nonzero_coefs': [1, 3, 6, 10],
                      'fit_intercept': [False, True],
                      'normalize': [False, True],
                      'precompute': [False, True]
                     }
        self.estimator_params['BayesianRidge'] = {
                      'alpha': [0.0000001, 0.00001, 0.001, 0.1],
                      'fit_intercept': [False, True],
                      'normalize': [False, True],
                      'precompute': [False, True]
                     }
        self.estimator_params['SGDRegressor'] = {
                      'alpha': [0.0000001, 0.00001, 0.001, 0.1],
                      'penalty': ['none', 'l2', 'l1', 'elasticnet'],
                      'fit_intercept': [False, True]
                     }
        self.estimator_params['HuberRegressor'] = {
                      'alpha': [0.0000001, 0.00001, 0.001, 0.1],
                      'epsilon': [1, 1.35, 2, 5],
                      'fit_intercept': [False, True]
                     }
        self.estimator_params['HuberRegressor'] = {
                      'alpha': [0.0000001, 0.00001, 0.001, 0.1],
                      'epsilon': [1, 1.35, 2, 5],
                      'fit_intercept': [False, True]
                     }
        self.estimator_params['RandomForestRegressor'] = {
                'n_estimators': [60, 100, 150],
                      'max_features': ['log2', 'sqrt','auto'],
                      'criterion': ['mse', 'mae'],
                      #'max_depth': [None, 8, 32, 64],
                      #'min_samples_split': [0.1, 0.2, 0.5, 0.7, 1.0],
                      #'min_samples_leaf': [1,2,5]
                     }
        self.estimator_params['GradientBoostingRegressor'] = {
                'n_estimators': [150],
                      #'loss': ['ls', 'lad','huber','quantile'],
                      #'criterion': ['mse', 'mae'],
                      #'max_depth': [None, 3, 5, 8],
                      #'min_samples_split': [0.2, 0.5, 0.9, 1.0],
                      #'min_samples_leaf': [1, 2, 3, 5],
                      #'learning_rate': [0.05, 0.1, 0.2, 0.3, 0.5],
                      #'alpha': [0.5, 0.7, 0.9, 1.0, 1.5]
                     }

    def set_logistic_regressors(self):
        rng = np.random.randint(100)
        self.estimators = [
            LogisticRegression(C=3),
            DecisionTreeClassifier(),
            GaussianNB(),
            MLPClassifier(alpha=1),
            QuadraticDiscriminantAnalysis(),
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            SVC(gamma=3, C=3),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            AdaBoostClassifier(),
            ExtraTreesClassifier(n_estimators=250,random_state=rng),
            IsolationForest(max_samples=100, random_state=rng)
        ]
        self.estimator_params = {}
        self.estimator_params['Logistic Regression'] = {'C': [1, 3, 6, 10],
                      'dual': [False],
                      'penalty': ['l2'],
                      'fit_intercept': [False, True],
                      'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                     }
        self.estimator_params['GaussianProcessClassifier'] = {'n_restarts_optimizer': [0, 1, 2],
                      'warm_start': [False, True],
                      'random_state': [None, 2]
                     }
        self.estimator_params['RandomForestClassifier'] = {'n_estimators': [4, 6, 9],
                      'max_features': ['log2', 'sqrt','auto'],
                      'criterion': ['entropy', 'gini'],
                      'max_depth': [2, 3, 5, 10],
                      'min_samples_split': [2, 3, 5],
                      'min_samples_leaf': [1,5,8]
                     }
        self.estimator_params['AdaBoostClassifier'] = {'n_estimators': [50, 25, 125],
                      'learning_rate': [0.5,1,2],
                      'algorithm': ['SAMME', 'SAMME.R'],
                      'random_state': [None, 2]
                     }
        self.estimator_params['ExtraTreesClassifier'] = {'n_estimators': [50, 150, 250],
                      'criterion': ['gini','entropy']
                     }
#        self.estimator_params['MSIASolver'] = {'max_iterations': [500, 2500, 5000],
#                      'learning_rate': [0.3,0.4,0.5]
#                     }
        
    def fit(self, estimator, test_ratio=0.1):
        self.detect_problem_type()
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.X)
        self.best_estimator = estimator
        self.best_estimator_name = estimator.__class__.__name__
        self.best_score = 0
        self.estimator_results = pd.DataFrame(index=range(1), columns=['name','score'])
        mean_outcome, std_outcome = self.run_kfold(estimator, estimator.__class__.__name__)
        self.fit_best(test_ratio)
        return self.accuracy, mean_outcome, std_outcome
        
    def fit_all(self):
        self.detect_problem_type()
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.X)
        #entrainement des différents classifieurs
        idx = 0
        self.best_score = 0
        self.estimator_results = pd.DataFrame(index=range(len(self.estimators)), columns=['name','score'])
        for estimator in self.estimators:
            self.run_kfold(estimator, estimator.__class__.__name__, idx)
            idx+=1
        print('Résultats des estimateurs'+"\n")
        self.estimator_results.sort_values(by='score', ascending=False, inplace=True)
        print(self.estimator_results,"\n")
        if len(self.estimator_results)>1:
            plt.figure()
            sns.barplot(x='score',y='name',data=self.estimator_results,palette="Set1")
            plt.show()
        print("\n"+'Best classifier = ',self.best_estimator_name)
        print('with score = ',self.best_score,"\n")
        return self
        
    def run_kfold(self, estimator, name, idx=0):
        kf = KFold(len(self.X), n_folds=15)
        outcomes = []
        fold = 0
        for train_index, test_index in kf:
            fold += 1
            X_train, X_test = self.X.values[train_index], self.X.values[test_index]
            y_train, y_test = self.Y.values[train_index], self.Y.values[test_index]
            X_train = self.scaler.transform(X_train)
            X_test = self.scaler.transform(X_test)
            #p('estimator',estimator)
            estimator.fit(X_train, y_train)
            predictions = estimator.predict(X_test)
            #accuracy = explained_variance_score(y_test, predictions)
            accuracy = estimator.score(X_test, y_test)
            outcomes.append(accuracy)
            print(name, " - Fold {0} accuracy: {1}".format(fold, accuracy),' RMSE : ',mean_squared_error(y_test, predictions))
        mean_outcome = np.mean(outcomes)
        std_outcome = np.std(outcomes)
        print("\n",name," - Mean Accuracy: {0}".format(mean_outcome),' +/- ',std_outcome)
        print("\n"+'---------------------------------------------------------'+"\n")
        if mean_outcome > self.best_score:
            self.best_score = mean_outcome
            self.best_estimator = estimator
            self.best_estimator_name = name
        self.estimator_results.at[idx,'name'] = name
        self.estimator_results.at[idx,'score'] = mean_outcome
        return mean_outcome, std_outcome
        
    def fit_best(self, test_ratio=0.1):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=test_ratio, random_state=0)
        # Type of scoring used to compare parameter combinations
        acc_scorer = make_scorer(explained_variance_score)
        # Run the grid search
        grid_obj = GridSearchCV(self.best_estimator, self.estimator_params[self.best_estimator_name], scoring=acc_scorer)
        
        
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        #X_train_columns = X_kaggle.columns
        #X_kaggle = self.scaler.transform(X_kaggle)  
        
        grid_obj = grid_obj.fit(self.X_train, self.y_train)
        # Set the estimator to the best combination of parameters
        self.best_estimator = grid_obj.best_estimator_
        #p('self.best_estimator',self.best_estimator)
        # Fit the best algorithm to the data.
        self.best_estimator.fit(self.X_train, self.y_train)
        #création des prédictions pour la validation intermédiaire
        predictions = self.best_estimator.predict(self.X_test)
        self.accuracy = explained_variance_score(self.y_test, predictions)
        print('Accuracy on test set = ', self.accuracy,"\n\n")
        #self.best_feats()
        return self
    
    def predict(self):
        return self.best_estimator.predict(self.X)
    
    def best_feats(self):
        X_train_columns = self.X.columns
        if hasattr(self.best_estimator, 'feature_importances_') and hasattr(self.best_estimator, 'estimators_'):
            importances = self.best_estimator.feature_importances_
            #print('importances',"\n",importances)
            indices = np.argsort(importances)[::-1]
            # Print the feature ranking
            print("\n","Feature ranking:")
            for f in range(self.n_dimensions-1):
                print("%d. %s (%f)" % (f + 1, X_train_columns[indices[f]], importances[indices[f]]))
            
            # Plot the feature importances of the forest
            plt.figure()
            plt.title("Feature importances")
            color = 'b'
            if hasattr(self.best_estimator.estimators_[0], 'feature_importances_'):
                std = np.std([tree.feature_importances_ for tree in self.best_estimator.estimators_], axis=0)
                plt.bar(range(self.n_dimensions), importances[indices], color=color, yerr=std[indices], align="center")
            else:
                plt.bar(range(self.X_train.shape[1]), importances[indices], color=color, align="center")
            plt.xticks(range(self.n_dimensions), X_train_columns[indices], rotation=45)
            plt.xlim([-1, self.n_dimensions])
            plt.show()
            
        if hasattr(self.best_estimator, 'coef_'):
            importances = self.best_estimator.coef_
            print('importances',type(importances),importances.shape,importances)
            indices = np.argsort(importances)[::-1]
            for f in range(len(X_train_columns)):
                print("%d. %s (%f)" % (f + 1, X_train_columns[indices[0,f]], importances[0,indices[0,f]]))
            
            # Plot the feature importances of the forest
            plt.figure()
            plt.title("Feature importances (relative)")
            #plt.bar(range(self.n_dimensions), importances[indices], color="r", yerr=std[indices], align="center")
            plt.bar(range(self.n_dimensions), importances[0,indices[0]], color="b", align="center")
            plt.xticks(range(self.n_dimensions), X_train_columns[indices[0]], rotation=45)
            plt.xlim([-1, self.n_dimensions])
            plt.show()
            
            # Plot the feature importances of the forest
            df_abs = pd.DataFrame(importances[0]).abs()
            indices = np.argsort(np.array([list(df_abs[0])]))
            df_abs_sorted = df_abs.sort_values(by=0,ascending= False)
            importances = np.array([list(df_abs_sorted[0])])
            plt.figure()
            plt.title("Feature importances (absolute)")
            #plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
            plt.bar(range(self.n_dimensions), importances[0,:], color="b", align="center")
            plt.xticks(range(self.n_dimensions), X_train_columns[indices[0][::-1]], rotation=45)
            plt.xlim([-1, self.n_dimensions])
            plt.show()
            
    def dump_result(self, X_predict, file_name='dump.csv'):
        self.X_predict = X_predict[self.select]
        self.Id_test = X_predict[self.id_col]
        predictions = self.best_estimator.predict(self.X_predict)
        output = pd.DataFrame({ self.id_col : self.Id_test, self.predict_column: predictions })
        output = output.set_index(self.id_col)
        output.to_csv(file_name)