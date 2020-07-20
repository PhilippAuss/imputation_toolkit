import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn import preprocessing
import time
from sklearn.metrics import r2_score

SEED = 42

"""
This python file contains functions for testing different imputation methods and compare them depending on their R2 score as well as visually in form of Qplots.
"""



def calc_perc_missing_values(df):
	"""
	Calculates the percentage of missing values per column of a dataframe
	and returns a dataframe with the columns and the corresponding missing value percentage.

	Parameters
	----------
	df: pandas dataframe

	Returns
	-------
	pandas dataframe with the percentage of missing values per attribute
	"""
    return df.isna().sum()/len(df)


def bar_plot_missing_values(df,figsize=(30,8),save_fig=False,file_name="bar_plot_missing_values"):
	"""
	Calculates the percentage of missing values per column of a dataframe
	and returns a dataframe with the columns and the corresponding missing value percentage.

	Parameters
	----------
	df: pandas dataframe

	Returns
	-------
	pandas dataframe with the percentage of missing values per attribute
	"""
    plt.style.use('ggplot')
    df_missing = df.isna().sum()/len(df)
    df_missing = df_missing.reset_index()
    df_missing = df_missing.rename({"index":"label",0:"percentage_missing"},axis=1)
    fig,ax = plt.subplots(figsize = figsize)
    df_missing.plot(kind="bar",ax = ax)
    plt.plot([0,len(df_missing)],[0.5,0.5],'k--',lw=2)
    plt.legend()
    fig.tight_layout()
    if(save_fig):
        plt.savefig(f"{file_name}.png")
    plt.show()
    return df_missing.sort_values(["percentage_missing"],ascending=True)


def QQ_matrixPlot(X, X_pred, column_names=[], fig_size = (15,10), title="Results", x_label = "actual values", y_label = "predicted values"):
	"""
	Plots the imputed values against the original values (ground truth)

	Parameters
	----------
	X: original full numpy matrix
	X_pred: numpy matrix with the imputed values
	column_names: list with the names of the attributes
	fig_size: tuple with the size of the output plot
	title: title string of the plot
	x_label: str
	y_label: str

	Returns
	-------
	ax: matplotlib axis element which can be ploted or stored
	"""
    plt.style.use('ggplot')
    n_obs, n_att = X_pred.shape

    if len(column_names) == 0:
        column_names = range(n_att)

    X = np.concatenate((X, X_pred), axis=0)
    X = preprocessing.minmax_scale(X)


    fig, ax = plt.subplots(figsize=(fig_size[0], fig_size[1]))
    ax.plot([0, 1], [0, 1], color='black', linestyle='dashed', linewidth=1)
    colors = iter(cm.rainbow(np.linspace(0, 1, n_att)))

    for col_name, i in zip(column_names, range(n_att)):
        ax.scatter(X[:n_obs, i], X[n_obs:, i], color=next(colors), label=col_name)

    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    fig.suptitle(str(title), fontsize = 20)
    fig.tight_layout()

    return ax


def binary_sampler(p, rows, cols, seed = SEED):
	'''
	Sample binary random variables.

	Parameters
	----------
	p: probability of 1
	rows: the number of rows
	cols: the number of columns
	seed: integer which indicates the random state

	Returns
	-------
	binary_random_matrix: generated binary random matrix.
	'''
    np.random.seed(seed)
    unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix


def _prepare_data(data_x, miss_rate=0.2, seed = SEED):
	'''
	replaces randomly values from a dataframe by NAs

	Parameters
	----------
	data_x: original dataframe without missing values
	miss_rate: rate of missing values which should be inserted
	seed: seed which is needed for reproducibility

	Returns
	-------
	miss_data_x: dataframe with missing values
	data_m: mask which indicates which entry is a missing value
	'''
    # Parameters
    no, dim = data_x.shape

    # Introduce missing data
    data_m = binary_sampler(1 - miss_rate, no, dim, seed)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

    return miss_data_x, data_m


def _test_imputation_method(method, imputer, X_missing, X, data_m, columns, R2_per_col = True,  seed = SEED):
	'''
	tests an imputation method on a numpy matrix

	Parameters
	----------
	method: name of the imputation method
	imputer: Imputer which should be tested, has to have the same interface as the sklearn IterativeImputers
	X_missing: numpy matrix with the data which should be imputed
	X: original data without missing values for comparison
	data_m: mask of the missing values
	columns: list of the column names of the X
	R2_per_col: boolean, which indicates if the R2 score for every singel column should be returned
	seed: set seed to achieve reproducibility	

	Returns
	-------
	df: dataframe with the results of the imputation test
	X_pred: numpy matrix with the imputed values
	'''    
    start_time = time.time()
    X_pred = imputer.fit_transform(X_missing)
    delta_time = [time.time() - start_time]
    df = pd.DataFrame()

    if(R2_per_col == True):
        for i, col in enumerate(columns):
            df[col] = [r2_score(y_true = ((1 - data_m) * X).T[i, :],y_pred = ((1-data_m)* X_pred).T[i,:])]



    # df["mse"] = ((X_pred[data_m] - X[data_m]) ** 2).mean()
    df["mean_R2"] = df[columns].mean(axis=1)
    df["method"] = method
    df["duration[s]"] = delta_time
    # print(df)

    return df, X_pred


def test_imputation_methods(df, imputer_dict, miss_rate=0.2, R2_per_col=True, seed = SEED,save_Qplot = False, filename ="Q_plot_best_model"):
	'''
	tests a dictionary of imputation methods on a numpy matrix

	Parameters
	----------
	df: Full dataframe, where imputation strategies should be compared
	imputer_dict: dictionary with imputation methods (sklearn IterativeImputer interface)
	miss_rate: rate of missing values, which are put into the df for testing purpose
	R2_per_col: boolean, which indicates if the R2 score for every singel column should be returned
	seed: set seed to achieve reproducibility	
	save_Qplot: boolean, which indicates if the Qplot of the best imputation strategy should be saved as png
	filename: name of the Qplot of the best imputer

	Returns
	-------
	df_results: dataframe with the imputation results for the different methods
	'''    
    X = df.values
    columns = df.columns

    X_missing, missing_mask = _prepare_data(X, miss_rate=miss_rate)

    results = []
    best_R2 = -np.inf
    X_best_R2 = None
    best_R2_method = None

    for method_name, imputer in tqdm(imputer_dict.items()):
        tqdm.write(f"Test {method_name}...")
        df_imp, X_pred = _test_imputation_method(method_name, imputer, X_missing, X, missing_mask, columns,R2_per_col, seed = seed)
        tqdm.write(f"finished")
        results.append(df_imp)

        if (df_imp["mean_R2"][0] > best_R2):
            best_R2 = df_imp["mean_R2"][0]
            X_best_R2 = X_pred
            best_R2_method = method_name

    df_results = pd.concat(results)
    Qplot = QQ_matrixPlot((1 - missing_mask) * X, (1 - missing_mask) * X_best_R2, column_names=columns, title = f"{best_R2_method}_(R2)")


    if(save_Qplot):
        plt.savefig(f"{best_R2_method}_(R2).png",pad_inches=0.3)

    plt.show()
    return df_results


def drop_cols_perc_na(df, perc_missing=0.5, cols_to_keep = []):
	'''
	drops columns of a dataframe if they have more than a specified percentage of missing values

	Parameters
	----------
	df: pandas dataframe
	perc_missing: rate of missing values, if higher column get's dropped
	cols_to_keep: list of columnames, which should be kept no matter how many missing values

	Returns
	-------
	df_res: resulting dataframe
	'''  
    n_samples, n_col = df.shape

    quote = int(n_samples * perc_missing)
    df_res = pd.DataFrame()

    for col in df.columns:
        n_missing = df[col].isna().sum()
        if (n_missing <= quote) or (col in cols_to_keep):
            df_res[col] = df[col]

    return df_res

