import pandas as pd
import numpy as np
import math
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge 
import os 
import warnings
warnings.simplefilter('ignore')

class LinearRegression(object):
    """The LinearRegression finds
     the best hypothesis class that can generalize well to new data and 
     can accurately predict the working-age population of the USA using polynomial curve-fitting for regression learning """
    def __init__(self,train_file, test_file):
        """init method will takes in train.dat and test.dat as inputs"""
        self.train_file = train_file
        self.test_file = test_file


    def normalization(self,col):
        """normalization is a method of rescaling input features so that they have a mean of 0 and 
        a standard deviation of 1"""
        mean_x = np.mean(col)
        std_x = np.std(col)
        norm_value = (col - mean_x)/std_x
        return norm_value

    def preprocessing(self):
        """ preprocessing method is done for both train and test dataset,this will get 
        normalised values and created the dataframe for the same"""
        colnames = ['year', 'population']
        df = pd.read_table(self.train_file, sep=' ', header=None, names=colnames)
        df_test = pd.read_table(self.test_file, sep=' ', header=None, names=colnames)
        norm_year_train = self.normalization(df['year'])
        norm_year_test = self.normalization(df_test['year'])
        get_norm_train = self.normalization(df['population'])
        get_norm_test = self.normalization(df_test['population'])
        df['year_norm'] = norm_year_train
        df_test['year_norm'] = norm_year_test
        df['population_norm'] = get_norm_train
        df_test['population_norm'] = get_norm_test
        return df, df_test


    def optimal_degree(self):
        """perform the optimal_degree for the given range till 12,
         we perform linear regression on 6 CV folds which returns best degree"""
        df,df_test = self.preprocessing()
        X = (df['year_norm'].values.reshape(-1,1))
        y = (df['population'])

        degrees = range(13)
        kfold = KFold(n_splits=6, shuffle=False, random_state=None)

        rmse_avg = []

        for degree in degrees:
            # generate polynomial features
            poly = PolynomialFeatures(degree)
            X_poly = poly.fit_transform(X)

            # initialize model
            model = Ridge(alpha=0)

            # perform cross-validation
            mse = []
            for train_index, test_index in kfold.split(X_poly):
                X_train, X_test = X_poly[train_index], X_poly[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                # fit model
                model.fit(X_train, y_train)

                # calculate mse on test set
                y_pred = model.predict(X_test)
                mse.append(mean_squared_error(y_test, y_pred))

            # calculate rmse for degree
            rmse_avg.append(np.mean(np.sqrt(mse)))
            # Find optimal degree based on minimum RMSE
            optimal_degree = degrees[np.argmin(rmse_avg)]

            # Print optimal degree
        print(f"Optimal degree: {optimal_degree}")
    

            # Compute and print average RMSE values for each degree
        for i, degree in enumerate(degrees):
            print(f"Degree {degree}: {rmse_avg[i]}")
            # # plot results
        return degrees, rmse_avg,optimal_degree

 

    def optimal_lambda(self):
        """perform optimal_lambda for the given values we perform linear regression
         on 6 CV folds which returns best lambda"""
        df,df_test = self.preprocessing()
        X = (df['year_norm'].values.reshape(-1,1))
        y = (df['population'])

                # define alpha and lambda values
        lambdas = [0, math.exp(-25), math.exp(-20), math.exp(-14), math.exp(-7), math.exp(-3), 1, math.exp(3), math.exp(7)]

        # set polynomial degree
        degree = 12

        rmse_mean = []

        for lam in lambdas:
            # generate polynomial features
            poly = PolynomialFeatures(degree)
            X_poly = poly.fit_transform(X.reshape(-1, 1))

            # initialize model
            model = Ridge(alpha=lam)

            # perform cross-validation
            rmse = []
            kfold = KFold(n_splits=6, shuffle=False, random_state=None)
            for train_index, test_index in kfold.split(X_poly):
                X_train, X_test = X_poly[train_index], X_poly[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                # fit model
                model.fit(X_train, y_train)

                # calculate rmse on test set
                y_pred = model.predict(X_test)
                rmse.append(np.sqrt(mean_squared_error(y_test, y_pred, squared=False)))   
                
            # calculate rmse for degree
            rmse_mean.append(np.mean(rmse))

        optimal_lambda = lambdas[np.argmin(rmse_mean)]
   
        print(f"Optimal lambda: {optimal_lambda}")
        print("Optimal lambda:e^({:.2f}) = {:.5f}".format(np.log(optimal_lambda), optimal_lambda))
        # Compute and print average RMSE values for each degree
        for i, lambd in enumerate(lambdas):
            print(f"Lamda {lambd}: {rmse_mean[i]}")

            
     
        return lambdas, rmse_mean, optimal_lambda

    def plots(self):
        """Plots returns plots for best degree and best lambda"""
        degrees, rmse_avg, od = self.optimal_degree()
        print(degrees,rmse_avg)
        fig, ax = plt.subplots()
        ax.plot(degrees, rmse_avg, marker='o')
        ax.set_xlabel('Degree')
        ax.set_ylabel('RMSE_AVERAGE')
        ax.set_title('RMSE vs Polynomial Degree')
        plt.show()

        lambdas, rmse_mean, optimal_lambda = self.optimal_lambda()
        fig, ax = plt.subplots()
        ax.plot(lambdas, rmse_mean, '-o')
        ax.set_xscale('log')
        ax.set_xlabel('Lambda')
        ax.set_ylabel('RMSE')
        plt.show()
        return ("end")

    def get_coeff_d(self):
        """get_coeff_d returns the coefficient for model when lambda is zero and degree is optimal"""
        _, _, d = self.optimal_degree()
        lambda_ = 0
        df,df_test = self.preprocessing()
        X = df['year_norm'].values.reshape(-1,1)
        y = df['population_norm']

        # Create polynomial features
        poly = PolynomialFeatures(degree=d)
        X_poly = poly.fit_transform(X)

        # Train the model using Ridge regression
        model = Ridge(alpha=lambda_)
        model.fit(X_poly,y)

        # Get the coefficient-weights
        coef_weights = model.coef_
        print("Coefficient weights at optimal degree",coef_weights)

        return ("coef_weights for degree is calculated")

    def get_coeff_lambda(self):
        """get_coeff_lambda method will returns coefficient of best lambda with degree 12 """
        _, _, lambda_ = self.optimal_lambda()
        degree = 12
        df,df_test = self.preprocessing()
        X = df['year_norm'].values.reshape(-1,1)
        y = df['population_norm']

        # Create polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)

        # Train the model using Ridge regression
        model = Ridge(alpha=lambda_)
        model.fit(X_poly,y)

        # Get the coefficient-weights
        coef_weights_l = model.coef_
        print("Coefficient weights at optimal lambda", coef_weights_l)

        return ("coef_weights for lambda is calculated")

    def train_rmse(self):
        """train_rmse method is used to calculate rmse values for training with best lambda and best degree """
        df,df_test = self.preprocessing()
        X = (df['year_norm'].values.reshape(-1,1))
        y = (df['population'])
        _, _, best_lambda = self.optimal_lambda()
        _, _, best_degree = self.optimal_degree()
        Xtrain_best = PolynomialFeatures(degree=12).fit_transform(X)
        ridge_reg_best = Ridge(alpha=best_lambda) 
        ridge_reg_best.fit(Xtrain_best, y) 
        ytrain_pred_best = ridge_reg_best.predict(Xtrain_best)
        rmse_train_best = np.sqrt(mean_squared_error(y, ytrain_pred_best))
        print("The rmse for training for best lambda : ",rmse_train_best)

        Xtrain_best = PolynomialFeatures(degree=best_degree).fit_transform(X)
        ridge_reg_best = Ridge(alpha=0) 
        ridge_reg_best.fit(Xtrain_best, y) 
        ytrain_pred_best = ridge_reg_best.predict(Xtrain_best)
        rmse_train_best_d = np.sqrt(mean_squared_error(y, ytrain_pred_best))
        print("The rmse for training for best degree : ",rmse_train_best_d)

        return("training rmse is calculated")

    def test_rmse(self):
        """test_rmse method is used to calculate rmse values for testing with best lambda and best degree"""
        df,df_test = self.preprocessing()
        Xtrain = (df['year_norm'].values.reshape(-1,1))
        ytrain = (df['population_norm'])
        X = (df_test['year_norm'].values.reshape(-1,1))
        y = (df_test['population_norm'])

        _, _,best_lambda = self.optimal_lambda()
        _, _, best_degree = self.optimal_degree()
        Xtrain_best = PolynomialFeatures(degree=12).fit_transform(Xtrain)
        ridge_reg_best = Ridge(alpha=best_lambda) # Create regression object
        ridge_reg_best.fit(Xtrain_best, ytrain) # Fit on regression object
        xtest = PolynomialFeatures(degree=12).fit_transform(X)
        ytrain_pred_best = ridge_reg_best.predict(xtest)
        rmse_train_best = np.sqrt(mean_squared_error(y, ytrain_pred_best))
        print("The rmse for testing for best lambda : ",rmse_train_best)

        Xtrain_best = PolynomialFeatures(degree=best_degree).fit_transform(Xtrain)
        ridge_reg_best = Ridge(alpha=0) # Create regression object
        ridge_reg_best.fit(Xtrain_best, ytrain) # Fit on regression object
        xtest = PolynomialFeatures(degree=best_degree).fit_transform(X)
        ytrain_pred_best = ridge_reg_best.predict(xtest)
        rmse_train_best_d = np.sqrt(mean_squared_error(y, ytrain_pred_best))
        print("The rmse for testing for best degree : ",rmse_train_best_d)

        return ("testing rmse is calculated")

    def age_range(self):
        """age_range method will return the 2 plots containing all the training data along with the
            resulting polynomial curves for d∗ and λ∗, for the range of years 1968-2023 as input  and for all lambdas"""
        df,df_test = self.preprocessing()

        X = df['year_norm'].values.reshape(-1, 1)
        y = df['population']

        # Set the degree of the polynomial
        degree = 6

        # Set the value of lambda
        alpha = 0

        # Generate polynomial features
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X)

        # Fit the model using Ridge regression
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_poly, y)

        # Define the range of years to plot
        years = np.linspace(1968, 2023, 100).reshape(-1, 1)
        years_norm = (years - df['year'].mean()) / df['year'].std()

        # Generate polynomial features for the range of years
        years_poly = poly.fit_transform(years_norm)

        # Predict the population for the range of years using the fitted model
        y_pred = ridge.predict(years_poly)

        # Plot the data and the fitted polynomial curve for the optimal degree
        fig, ax = plt.subplots()
        ax.scatter(df['year'], df['population'], s=10, label='Training Data')
        ax.plot(years, y_pred, label='Prediction')
        ax.set_xlabel('Year')
        ax.set_ylabel('Population')
        ax.set_title(f'Ridge Regression for Degree {degree} and Lambda {alpha:.1f}')
        ax.legend()
        plt.show()

        # Generate polynomial features for the range of lambdas
        lambdas = [0, math.exp(-25), math.exp(-20), math.exp(-14), math.exp(-7), math.exp(-3), 1, math.exp(3), math.exp(7)]

        lambda_poly = poly.fit_transform(X)

        # Store the coefficients for each lambda value
        coef_list = []
        for lambd in lambdas:
            ridge = Ridge(alpha=lambd)
            ridge.fit(lambda_poly, y)
            coef_list.append(ridge.coef_)

        return ("age_range executed")

    def age_range_d(self):
        df,df_test = self.preprocessing()

        X = df['year_norm'].values.reshape(-1, 1)
        y = df['population'].values

        # Set the degree of the polynomial
        degree = 12

        # Set the value of lambda
        _, _, alpha = self.optimal_lambda()

        # Generate polynomial features
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X)

        # Fit the model using Ridge regression
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_poly, y)

        # Define the range of years to plot
        years = np.linspace(1968, 2023, 100).reshape(-1, 1)
        years_norm = (years - df['year'].mean()) / df['year'].std()

        # Generate polynomial features for the range of years
        years_poly = poly.fit_transform(years_norm)

        # Predict the population for the range of years using the fitted model
        y_pred = ridge.predict(years_poly)

        # Plot the data and the fitted polynomial curve for the optimal degree
        fig, ax = plt.subplots()
        ax.scatter(df['year'], df['population'], s=10, label='Training Data')
        ax.plot(years, y_pred, label='Prediction')
        ax.set_xlabel('Year')
        ax.set_ylabel('Population')
        ax.set_title(f'Ridge Regression for Degree 12 and Lambda {alpha}')
        ax.legend()
        plt.show()

        # Generate polynomial features for the range of lambdas
        # lambdas = math.exp(-3)
        # lambda_poly = poly.fit_transform(X)

        # # Store the coefficients for each lambda value
        # coef_list = []

        # ridge = Ridge(alpha=lambdas)
        # ridge.fit(lambda_poly, y)
        # coef_list.append(ridge.coef_)

        return (" age_range method will return the 2 plots containing all the training data along with the resulting polynomial curves for d∗ and λ∗, for the range of years 1968-2023 as input  and for all lambdas")

    def call_all(self):
        res1 = self.optimal_degree()
        res2 = self.optimal_lambda()
        res9 = self.plots()
        res3 = self.get_coeff_d()
        res4 = self.get_coeff_lambda()
        res5 = self.train_rmse()
        res6 = self.test_rmse()
        res7 = self.age_range()
        res8 = self.age_range_d()
        return (res1,res2,res9,res3, res4, res5,res6, res7, res8)



if __name__ == '__main__':
    inputfile = 'train.dat'
    testfile = 'test.dat'
    model = LinearRegression(inputfile, testfile)
    res = model.call_all()
    print(res)








    

    

