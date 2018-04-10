import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cross_validation import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, explained_variance_score, confusion_matrix, mean_squared_error


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
        hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols.values, xticklabels=cols.values)
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
    
    def auto_clean(self, df=None, predict_column=None):
        if df is None:
            df = self.X
        else:
            self.X = df.copy()
        if predict_column is None:
            predict_column = self.predict_column
        else:
            self.predict_column = predict_column
        self.X = df.drop(predict_column, axis=1).copy()
        self.n_dimensions = df.shape[1]
        self.n_samples = df.shape[0]
        self.Y = df[predict_column].copy()
        self.columns = df.columns
#        print('auto_clean:')
#        print('n_dimensions:',self.n_dimensions)
#        print('n_samples:',self.n_samples)
#        print('range_x:',self.range_x)
#        print('columns:',self.columns)
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
#                print('col',col)
#                print('unique',unique)
#                print('means',means)
#                print('indices',indices)
#                print('indices2',indices2)
#                print('map_array',map_array)
                df[col] = df[col].map(map_array)
        self.X = df.drop(predict_column, axis=1).copy()
        self.range_x = np.max(np.abs(self.X))
        return df, mapping
    
    def map_clean(self, df, mapping):
        for col in df.columns:
            col_type = df[col].dtype
            if col_type == 'object' and col in mapping:
                map_array = mapping[col]
                df[col] = df[col].map(map_array)
        return df
        
    def detect_problem_type(self):
        """Solver: detect problem type
        """
        if self.problem_type == None:
            if(self.Y.shape[1]==1):
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
            LinearRegression(),
            Ridge(),
            RidgeCV(),
            Lasso(),
            MultiTaskLasso(),
            ElasticNet(),
            ElasticNetCV(),
            MultiTaskElasticNet(),
            Lars(),
            LassoLars(),
            OrthogonalMatchingPursuit(),
            BayesianRidge(),
            ARDRegression(),
            SGDRegressor(),
            PassiveAggressiveRegressor(),
            HuberRegressor(),
            RandomForestRegressor(),
            GradientBoostingRegressor()
        ]

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
        
    def fit(self):
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.X)
        
    def fit_all(self):
        #entrainement des différents classifieurs
        idx = 0
        best_score = 0
        estimator_results = pd.DataFrame(index=range(len(self.estimators)), columns=['name','score'])
        for estimator in self.estimators:
            best_clf, best_score, best_clf_name = self.run_kfold(estimator, estimator.__class__.__name__, best_score, estimator_results, idx)
            idx+=1
        
    def run_kfold(self, estimator, name, best_score, estimator_results, idx):
        kf = KFold(len(self.X), n_folds=15)
        outcomes = []
        fold = 0
        for train_index, test_index in kf:
            fold += 1
            X_train, X_test = self.X.values[train_index], self.X.values[test_index]
            y_train, y_test = self.Y.values[train_index], self.Y.values[test_index]
            X_train = self.scaler.transform(X_train)
            X_test = self.scaler.transform(X_test)
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
        if mean_outcome > best_score:
            best_score = mean_outcome
            best_estimator = estimator
            best_estimator_name = name
        estimator_results.at[idx,'name'] = name
        estimator_results.at[idx,'score'] = mean_outcome
        return (best_estimator, best_score, best_estimator_name)