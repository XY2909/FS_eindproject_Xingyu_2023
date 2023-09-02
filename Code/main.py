import pandas as pd
import numpy as np
import glob
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from joblib import dump, load
import tqdm
from scipy.stats import shapiro
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from tqdm import tqdm
import xgboost as xgb
from matplotlib import pyplot

def load_files():
    # this function load all the csv files and 
    # concatenate them together to form a large dataframe

    # create an empty list to store the dataframes
    dfs = []

    # define path where all the csvs are saved
    # in this case in the local folder called 'IM Meetingen 2021'
    csvs = glob.glob(__path__)

    # loop through all the csv files and store them in a list
    for file in csvs:
        df = pd.read_csv(file, on_bad_lines = 'skip',  sep = ";")
        dfs.append(df)

    # concatenate the dataframes in the list into one big dataframe
    df = pd.concat(dfs, ignore_index=True)

    df.to_csv(r"C:\Users\panxi\RP\raw_2019.csv", sep = ';', mode = 'w')


def step1():
    # create an empty list to store the dataframes
    dfs_clean = []

    # define path where all the csvs are saved
    csvs = glob.glob(__path__)

    # loop through all the csv files and store them in a list
    for file in csvs:
        df = pd.read_csv(file, on_bad_lines = 'skip',  sep = ";", skiprows=[0], low_memory=False)

        df = handle_nans(df)
        
        dfs_clean.append(df)
    
    df1 = dfs_clean[0]
    df2 = dfs_clean[1]
    df3 = dfs_clean[2]
    
    # Find the common columns between the dataframes
    common_cols_1_2 = list(np.intersect1d(df1.columns, df2.columns))

    # Merge the dataframes one by one
    merged_df12 = pd.merge(df1, df2, how = 'outer', on = common_cols_1_2)

    common_cols_1_2_3 = list(np.intersect1d(merged_df12.columns, df3.columns))

    merged_df123 = pd.merge(merged_df12, df3, how='outer', on=common_cols_1_2_3)

    merged_df123.to_csv(__path__, sep = ';', mode = 'w')
    
def combine_all():
    # create an empty list to store the dataframes
    dfs = []

    # define path where all the csvs are saved
    csvs = glob.glob(__path__)

    # loop through all the csv files and store them in a list
    for file in csvs:
        df = pd.read_csv(file, on_bad_lines = 'skip',  sep = ";", skiprows=[0], low_memory=False)
        dfs.append(df)

    df1 = dfs[0]
    df2 = dfs[1]
    #df3 = dfs[2]

    # Find the common columns between the dataframes
    common_cols_1_2 = list(np.intersect1d(df1.columns, df2.columns))

    # Merge the dataframes one by one
    merged_df12 = pd.merge(df1, df2, how = 'outer', on = common_cols_1_2)

    #common_cols_1_2_3 = list(np.intersect1d(merged_df12.columns, df3.columns))
    
    #merged_df123 = pd.merge(merged_df12, df3, how='outer', on=common_cols_1_2_3)

    merged_df12.to_csv(__path__, sep = ';', mode = 'w')

    return merged_df12

def drop_irrelevant(df):

    # 1. DROP FISH: drop every rows denoting fish caught
    # get names of indexes for which column Parameter.type has value Biotaxon
    try:
        index_fish = df[df['Parameter.type'] == 'Biotaxon' ].index
    except KeyError:
        index_fish = df[df['Parameter.groep'] == 'Biotaxon' ].index
    
    # drop these row indexes from the dataFrame
    df.drop(index_fish, inplace = True)

    # 2. DROP NO ID: drop everything where sample ID is none
    df.dropna(subset=['Monster.Identificatie'], inplace=True)

    ###### get rows matching monsterID_loc_date ########
    # df_s = pd.read_excel(__path__)
    # monsterIDs = df_s['MonsterID__Loc__date'].tolist()


    # mask = df['MonsterID__Loc__date'].isin(monsterIDs)

    # # Apply the mask to select the rows matching the values in column 'A'
    # filtered_df = df[mask]
    
    ##################################################
    # 3. GET PFOA only: 
    # get the Monster.Identificatie (sample identification code) 
    # when PFOA is measured, drop every other samples where no PFOA is detected

    #all_pfoas = df[(df['Parameter.code'] == 'PFOA') & (df['Numeriekewaarde'] == 0)] 

    # all_pfoas = all_pfoas[(all_pfoas['Numeriekewaarde'] == 0)]

    # print('all_pfoas shape: ', all_pfoas.shape)

    # #index_code = list(all_pfoas['Monster.Identificatie'])  # strings in a list
    #index_code = list(all_pfoas['MonsterID__Loc__date'])

    # print('length index code: ', len(index_code))

    # # NORMAL: now selected rows from the original dataframe
    # # df_new: samples where PFOAs (and or PFOS) are measured

    # df_new = df.loc[df['MonsterID__Loc__date'].isin(index_code)]

    # print('shape df_new: ', df_new.shape)

    # MODIFICATION: select all rows where no PFOA is measured
    # exclude_indices = index_code

    # df_new = df.loc[~df['MonsterID__Loc__date'].isin(exclude_indices)]


    return df

def show_stat(df):
    # total samples during jan-feb -> 673
    print('# samples', len(df["Monster.Identificatie"].unique())) 

    # count the occurence of PFOA/PFOS in dataset -> 673 times measured (equal to number of samples)
    print('times PFOA measured: ', df['Parameter.code'].value_counts()['PFOA']) 

    # number of unique substances: 586
    print('unique substances: ', len(df["Parameter.code"].unique()))

    print('number of columns: ', len(df.columns))

def convert_to_micrograms(row):
    # convert miligram to microgram
    if row['Eenheid.code'] == 'mg/l':
        row['new_values'] = float(row['Numeriekewaarde']) * 1000
        row['new_eenheid'] = 'ug/l'

    # convert nanogram to microgram 
    elif row['Eenheid.code'] == 'ng/l':
        row['new_values'] = float(row['Numeriekewaarde']) / 1000
        row['new_eenheid'] = 'ug/l'

    # keep original units
    else:
        row['new_values'] = float(row['Numeriekewaarde'])
        row['new_eenheid'] = row['Eenheid.code']
    
    return row

def join_columns(row):
    # this function joins three columns together by dubble underscores;
    # if 'parameter.code' is empty, it is probably a unit instead of substance,
    # in that case the name is picked from the 'Grootheid.code' column.

    # if pd.isnull(row['Parameter.code']):
    #     return f"{row['Grootheid.code']}__{row['Hoedanigheid.code']}__{row['new_eenheid']}"
    
    # return f"{row['Parameter.code']}__{row['Hoedanigheid.code']}__{row['new_eenheid']}"

    if pd.isnull(row['Parameter.code']):
        return f"{row['Grootheid.code']}__{row['Hoedanigheid.code']}__{row['Eenheid.code']}"
    
    return f"{row['Parameter.code']}__{row['Hoedanigheid.code']}__{row['Eenheid.code']}"

def wide_to_long(df):

    # set the row values of Numeriekewaarde to zero where Limietsymbool is '<'
    df.loc[df['Limietsymbool'] == '<', 'Numeriekewaarde'] = 0

    # add a new column to the dataframe 
    df['MonsterID__Loc__date'] = df[['Monster.Identificatie', 'Meetobject.code', 'Begindatum']].apply(lambda x: '__'.join(x.dropna().astype(str)), axis=1)

    # convert time to datetime format (e.g., 01-01-2021), 
    # then get the week of the year from this date
    df['date'] = pd.to_datetime(df['Begindatum'], format='%Y-%m-%d', errors='coerce').fillna(pd.to_datetime(df['Begindatum'], format='%d-%m-%Y', errors='coerce'))

    df['week'] = df['date'].dt.isocalendar().week

    # create a new column from three existing columns
    df['Parameter__Hoedanigheid__Eenheid'] = df.apply(join_columns, axis=1)

    # Remove excess data
    df = drop_irrelevant(df)
    
    

    # convert all units to ug/l if possible 
    # df = df.apply(convert_to_micrograms, axis=1)

    # finally, convert wide format to long format
    return pd.pivot_table(df,  index = ['week', 'MonsterID__Loc__date'], columns = ['Parameter__Hoedanigheid__Eenheid'], values=['Numeriekewaarde'], aggfunc = lambda x:x)

def remove_cols_with_zero(df):
    # This function removes cols in df if more than 
    # 95% of the column values are zeros

    number_of_rows = df.shape[0]

    # Calculate the percentage of NaN and zero values in each column
    nan_percent = df.isna().sum() / number_of_rows * 100
    zero_percent = df.eq(0).sum() / number_of_rows * 100

    # Calculate the combined percentage of NaN and zero values in each column
    combined_percent = nan_percent + zero_percent

    # Identify the columns with a combined percentage more than 95%
    high_zero_nans_cols = combined_percent[combined_percent > 95].index.tolist()

    # Remove the high zero-nan columns from the DataFrame
    df.drop(high_zero_nans_cols, axis=1, inplace=True)

    return df

def handle_nans(df):

    # 1. Remove every column with only NaNs and zeros
    # 1.1 find those columns and store them in a list 
    zero_nan_cols = [col for col in df.columns if not any(df[col].dropna().unique())]

    # 1.2 drop those columns 
    df = df.drop(columns = zero_nan_cols)
    
    # 1.3 drop cols with high percentages of zeros
    df = remove_cols_with_zero(df)

    #print('shape of df after removing high zero cols', df.shape)

    # 2. Now iteratively drop irrelevant columns and rows
    # until the dataset is complete (i.e. no NaNs present)

    while (df.isna().sum().max() > 0):

        # Calculate the number of missing values in each row and column
        NArows = df.isna().sum(axis=1)
        NAcols = df.isna().sum(axis=0)

        # Find the row and column with the maximum number of missing values
        mNArow = NArows.idxmax()
        mNAcol = NAcols.idxmax()

        # Calculate the proportion of missing values in the row and column with the maximum number of missing values
        mNArow_prop = NArows[mNArow] / df.shape[1]
        mNAcol_prop = NAcols[mNAcol] / df.shape[0]

        # Remove the row or column with the highest proportion of missing values
        if mNArow_prop > mNAcol_prop:
            # remove 1 row
            df.drop(mNArow, axis=0, inplace=True)


        else:
            # remove 1 column
            df.drop(mNAcol, axis=1, inplace=True)
           


    return df


def randomized_search(X_train, y_train):
    hyperparameters = {
    'n_estimators': np.arange(100, 1000, 100),
    'max_depth': [None] + list(np.arange(5, 30, 5)),
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 10],
    'max_features': ['auto', 'sqrt', 'log2']}

    rf = RandomForestRegressor()

    # Define the random search
    random_search = RandomizedSearchCV(
    rf, 
    param_distributions=hyperparameters, 
    n_iter=50, 
    cv=10, 
    scoring='neg_mean_squared_error')

    # Fit the random search to the training data
    random_search.fit(X_train, y_train)

    # print the best hyperparameters and their score
    print('Best hyperparameters:', random_search.best_params_)
    print('Best score:', random_search.best_score_)

    return random_search.best_params_


def randomized_search_xgboost(X_train, y_train, X_test, y_test):
    
    # Define the XGBoost regression model
    model = xgb.XGBRegressor()

    # Define the parameter grid for the randomized search
    param_grid = {
        'n_estimators': np.arange(100, 1000, 100),
        'learning_rate': [0.1, 0.01, 0.001],
        'max_depth': [3, 5, 6, 7, 8],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': np.arange(0, 1)
    }

    # Perform randomized search with cross-validation
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=50,              # Number of parameter settings to sample
        scoring='neg_mean_squared_error',  # Use negative mean squared error as the evaluation metric
        cv=10,                   # Number of cross-validation folds
        random_state=42         # Random seed for reproducibility
    )

    # Fit the randomized search to the training data
    random_search.fit(X_train, y_train)

    # Get the best model and its hyperparameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    # Make predictions on the test set using the best model
    y_pred = best_model.predict(X_test)

    # Evaluate the best model
    r2 = r2_score(y_test, y_pred)

    print("Best Model R-squared:", r2)
    print("Best Hyperparameters:", best_params)

    return best_params

def baseline_model(X_train, y_train, X_test, y_test):
    # Create a dummy model that always predicts the mean value of the target variable
    dummy = DummyRegressor(strategy='mean')

    # Train the dummy model on the training data
    dummy.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = dummy.predict(X_test)

    # Calculate the mean squared error of the dummy model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    
    # 3. Take the square root of the result to get Squared Root of MSE (RMSE)
    rmse = np.sqrt(mse)
    std_dev = np.std(y_test)
    #std_rms = rmse/std_dev

    print('Mean squared error of dummy model:', mse)
    print('R2 of dummy model:', r2)
    print('RMSE: ', rmse)

def RFR(X_train, y_train):
    
    feature_names = X_train.columns 

    tuned_hyperparameters = randomized_search(X_train, y_train)
    # obtained from random search 

    #tuned_hyperparameters = {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 10}
    #tuned_hyperparameters = {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 15}
    # Initialize the model
    rf = RandomForestRegressor(**tuned_hyperparameters)


    mse_score = -cross_val_score(rf, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    r2_score = cross_val_score(rf, X_train, y_train, cv=10, scoring='r2')

    # Fit the model on the entire dataset
    rf.fit(X_train, y_train)

    path = str
    
    # Store the trained model using joblib
    # example: 'C:\Users\...\model.joblib'
    dump(rf, path)

    # calculate the mean and standard deviation of the scores
    mean_mse = mse_score.mean()
    std_mse = mse_score.std()
    mean_r2 = r2_score.mean()
    std_r2 = r2_score.std()

    print('Mean squared error:', mean_mse)
    print('Standard deviation of mean squared error:', std_mse)
    print('R2 score:', mean_r2)
    print('Standard deviation of R2 score:', std_r2)

    print('MSE scores from 5 folds:', mse_score)
    print('R2 scores from 5 folds:', r2_score)

    # get feature importance from the trained RFR model 
    importances = rf.feature_importances_  

    # create a pandas DataFrame with feature names and their importance scores
    df_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # sort the DataFrame by importance scores in descending order
    df_importances = df_importances.sort_values(by='Importance', ascending=False)

    # keep only the top 20 features
    df_importances = df_importances.iloc[:20]

    print('the top 20 most important features are: \n', df_importances)

    # set the 'Feature' column as the index
    df_importances = df_importances.set_index('Feature')

    # plot the feature importances
    ax = df_importances.plot(kind='bar', figsize=(8,6))

    # rotate x-axis tick labels
    plt.xticks(rotation=40, ha='right', size = 5)

    ax.set_ylabel('Importance')
    ax.set_title('Feature Importance Ranking after log transformation')
    
    path_to_figure = str
    # save figure
    plt.savefig(path_to_figure)

def model_eval(X_test, y_test):

    # Load the trained model
    rf_loaded = load(__path__)

    # Use the loaded model for prediction
    y_pred = rf_loaded.predict(X_test)

    # Evaluate predictions using MSE and R-squared (in log transformed form)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("scaled MSE test set:", mse)
    print("R-squared test set:", r2)

    # Get more interpretable evaluation:

    # 1. transform back predictions in original format
    epsilon = 1e-6
    y_pred_back = np.exp(y_pred) - epsilon

    # 2. Get the MSE between these invert-transformed predictions and the original data
    y_test_back = np.exp(y_test) - epsilon

    real_mse = mean_squared_error(y_test_back, y_pred_back)

    # 3. Take the square root of the result to get Squared Root of MSE (RMSE)
    rmse = np.sqrt(real_mse)

    # 4. Determine the range of the target variable
    # and calculate the error rate: it will represent the average error as a percentage of the data range.
    data_range = np.max(y_test_back) - np.min(y_test_back)
    error_rate = (rmse / data_range) * 100

    # Additional: Calculate Mean Absolute Percentage error (MAPE)
    # Attention: the output can be arbitrarily high when y_true is small (which is specific to the metric) or when abs(y_true - y_pred) is large
    MAPE = np.mean(np.abs((y_test_back - y_pred_back) / y_test_back)) * 100

    # MPE: mean percentage error
    # negatives mean predictions are systematically underestimated. Positive means otherwise
    MPE = np.mean((y_test_back - y_pred_back) / y_test_back)

    print(f'RMSE testset: {rmse}')

    print(f'The relative error is approx. {error_rate}%, MAPE is {MAPE}%, MPE: {MPE}')


    # create a scatterplot of Y_test and Y_pred with a regression line
    # and 95% confidence interval
    sns.regplot(x = y_test, y = y_pred, ci = 95, label = 'Model prediction')

    # # add a line for perfect prediction
    plt.plot(y_test, y_test, color = 'red', label = 'Golden standard')

    # set x and y axis labels
    plt.xlabel('log PFOA (µg/L)')
    plt.ylabel('log PFOA (µg/L)')

    # Add a legend  
    plt.legend()   

    # Add a title
    plt.title(label = 'XGboost evaluation')

    # save the plot as a png file
    plt.savefig(__path__)
    
def permutation(clf, X_train, y_train):

    feature_names = X_train.columns 
    result = permutation_importance(clf, X_train, y_train, n_repeats=10, random_state=42)
    perm_sorted_idx = result.importances_mean.argsort()


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    tree_importance_sorted_idx = np.argsort(clf.feature_importances_)
    tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5


    top_k_features = 30  # Specify the number of top features to display

    df_importances = pd.DataFrame(result)

    # sort the DataFrame by importance scores in descending order
    df_importances = df_importances.sort_values(by='Importance', ascending=False)

    top_features = feature_names[perm_sorted_idx][-top_k_features:]  # Get the top k features
    top_importances = result.importances[perm_sorted_idx][-top_k_features:]  # Get the importances for the top k features

    top_importances_mean = result.importances_mean[perm_sorted_idx][-top_k_features:]  # Get the importances for the top k features


    print(top_features.shape, top_importances.shape, top_importances_mean.shape)

    ax1.barh(range(len(top_features)), top_importances_mean, height=0.7)

    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features)
    ax1.set_xlabel('Importance')
    #ax1.set_ylabel('Feature')
    ax1.set_title(f'Top {top_k_features} Feature Importances')

    ax2.boxplot(top_importances.T, vert=False, labels=top_features)  # Plot the top k features and their importances
    ax2.set_title(f'Permutation importances of the top {top_k_features} features')

    fig.tight_layout()
    
    # plt.title(label = '')

    # save the plot as a png file
    plt.savefig(__path__)



def log_transformation(X_train, X_test):

    modified_features = []

    for feature in X_train.columns:
        # Test normality before log transformation
        stat, p = shapiro(X_train[feature])

        #print('Normality test before log transformation: feature', feature, 'stat=%.3f, p=%.3f' % (stat, p))
        
        # Test normality after log transformation
        stat, p = shapiro(np.log(X_train[feature]))
        #print('Normality test after log transformation: feature', feature, 'stat=%.3f, p=%.3f' % (stat, p))

        
        # Apply log transformation if normality improves
        if p > 0.05:
            X_train[feature] = np.log(X_train[feature]+ 1e-8)
            X_test[feature] = np.log(X_test[feature]+ 1e-8)
            modified_features.append(feature)
        
    print('modified features: \n', modified_features)

    return X_train, X_test


def handle_multicollinear(X):

    feature_names = X.columns
    

    # ############## DENDROGRAM ###############
    ax1 = plt.gca()
    corr = spearmanr(X).correlation

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # # We convert the correlation matrix to a distance matrix before performing
    # # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=feature_names.tolist(), ax=ax1 , leaf_font_size= 8)
    

    plt.axis('tight')
    plt.tight_layout()

    plt.savefig(__path__)



    ########### HEATMAP ############
    # plt.show()
    # fig, ax2 = plt.subplots(figsize=(10, 8))
    
    # dendro_idx = np.arange(0, len(dendro["ivl"]))
    # ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])


    # ax2.set_xticks(dendro_idx)
    # ax2.set_yticks(dendro_idx)
    # ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    # ax2.set_yticklabels(dendro["ivl"])

    # ax2.tick_params(axis='both', labelsize=5)

    # fig.tight_layout()
    #plt.show()
    #plt.savefig()


    # Remove features from clusters
    # manually pick a threshold: 2
    temp = {}
    for t in np.arange(0.5, 4, 0.5):

        cluster_ids = hierarchy.fcluster(dist_linkage, t, criterion="distance")
        cluster_id_to_feature_ids = defaultdict(list)
        cluster_id_to_feature_names = defaultdict(list)

        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)

            cluster_id_to_feature_names[cluster_id].append(feature_names[idx])
        
        # print(cluster_id_to_feature_ids)
        print(cluster_id_to_feature_names)

        # This line creates a list of selected features by picking A RANDOM index from the list of feature indices for each cluster ID. 
        # This essentially selects a representative feature for each cluster.
        selected_features = [random.choice(v) for v in cluster_id_to_feature_names.values()]
    
    
        print(f'threshold: {t};  number of selected features: {len(selected_features)}')
        print(selected_features)

def adjusted_R(R, n, p):
    
    adjusted_R_square = 1 - (1-R)*(n-1)/(n-p-1)

    return adjusted_R_square

def experiment_knmi():

    # force read_excel to treat empty cells as NaNs instead of empty strs
    df_knmi = pd.read_excel(r'C:\Users\panxi\RP\chem_knmi.xlsx')
    
    # replace spaces with NaN values
    df_knmi = df_knmi.replace('^\s*$', np.nan, regex=True)
    
    #print(df_knmi.isna().sum())

    print('shape df chem_knmi(row, col) before removing nans: ', df_knmi.shape)

    df = handle_nans(df_knmi)

    print('shape df chem_knmi(row, col) after removing nans: ', df.shape)

    df.to_csv(r'C:\Users\panxi\RP\chem_knmi_removedN.csv',  sep = ';', mode = 'w')



def xgboost(X_train, y_train, X_test, y_test):

    exp_number = 1

    feature_names = X_train.columns 
    
    tuned_hyperparameters = randomized_search_xgboost(X_train, y_train, X_test, y_test)
    
    # Define the XGBoost regression model
    model = xgb.XGBRegressor(**tuned_hyperparameters)

    # # # Train the model
    model.fit(X_train, y_train)
    
    mse_score = -cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    r2_score = cross_val_score(model, X_train, y_train, cv=10, scoring='r2')

    # calculate the mean and standard deviation of the scores
    mean_mse = mse_score.mean()
    std_mse = mse_score.std()
    mean_r2 = r2_score.mean()
    std_r2 = r2_score.std()

    print('Mean squared error:', mean_mse)
    print('Standard deviation of mean squared error:', std_mse)
    print('R2 score:', mean_r2)
    print('Standard deviation of R2 score:', std_r2)

    print('MSE scores from 5 folds:', mse_score)
    print('R2 scores from 5 folds:', r2_score)

    # get feature importance from the trained RFR model 
    importances = model.feature_importances_  

    # create a pandas DataFrame with feature names and their importance scores
    df_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # sort the DataFrame by importance scores in descending order
    df_importances = df_importances.sort_values(by='Importance', ascending=False)

    # keep only the top 20 features
    df_importances = df_importances.iloc[:20]

    print('Important features are: \n', df_importances)
    
    #set the 'Feature' column as the index
    df_importances = df_importances.set_index('Feature')

    # plot the feature importances
    ax = df_importances.plot(kind='bar', figsize=(8,6))

    # rotate x-axis tick labels
    plt.xticks(rotation=40, ha='right', size = 5)

    ax.set_ylabel('Importance')
    ax.set_title('Feature Importance Ranking after log transformation')

    # save figure
    plt.savefig(__path__)

def main():

    # experiment_knmi()
    # step()
    # #load_files()
    # df = combine_all()

    # df = pd.read_excel(__path__)

    # df = df.replace(',', '.', regex=True)
    # df = df.replace(r'^\s*$', np.nan, regex=True)

    # print('shape df complete 20-21 before (row, col): ', df.shape)

    # df = handle_nans(df)

    # print('shape df complete 19-21 after removing nans (row, col): ', df.shape)


    # # Convert wide format to long format
    # # Note: round(10) is needed for float precision (for writing to csv)! To prevent bug !!
    
    # df = wide_to_long(df).round(10)

    # # df2019_raw_long = wide_to_long(df)

    # SKIP ROW: first row need to be skipped to prevent error 


    ### Handle missing values ### 
    # df_long_complete = handle_nans(df_long)

    # Describe pfoa stats
    # print(df_long_complete['PFOA__NVT__ug/l'].describe())

    ############ TRAIN MODEL ###############


    # # print('shape row, col: ', df.shape)

    df_long_complete = pd.read_excel(__path__)

    df_long_complete = df_long_complete.replace(',', '.', regex=True)
    df_long_complete = df_long_complete.replace(r'^\s*$', np.nan, regex=True)

    # # # df_long_complete = handle_nans(df_long_complete)
    
    # print('shape df (row, col): ', df_long_complete.shape)
    
    # # ignore the first 4 columns
    df_long_complete = df_long_complete.iloc[:, 5:] 

    # some numerical values are separated by comma, replace them with dot
    df_long_complete = df_long_complete.replace(',', '.', regex=True)

    # Replace string values with NaN and cast columns to float
    df_long_complete = df_long_complete.replace('', float('nan')).astype(float)



    y = df_long_complete['PFOA__NVT__ug/l']

    X = df_long_complete.drop('PFOA__NVT__ug/l', axis=1)
    X = X.drop('log pfoa', axis=1)
    X = X.drop('STN', axis=1)
    #X = X.drop('PFOA__NVT__ug/l', axis=1)

    # ############# REMOVE ALL PFAS #############
    
    # pfas_list = ['L_PFBS__NVT__ug/l', 'L_PFHpS__NVT__ug/l', 'L_PFHxS__NVT__ug/l', 'PFBA__NVT__ug/l', 'PFC5asfzr__NVT__ug/l', 'PFDA__NVT__ug/l', 'PFHpA__NVT__ug/l', 'PFHxA__NVT__ug/l', 'PFNA__NVT__ug/l', 'PFOSA__NVT__ug/l', 'PFPA__NVT__ug/l', 'sverttPFHxS__NVT__ug/l', 'sverttPFOS__NVT__ug/l', 'N-MeFOS', 'FOSAA', 'FRD']
    
    top5 = ['Ni__nf__ug/l', 'O2__NVT__%', 'DR', 'Dist_to_RWI', 'UN']

    X = X[top5]

    # filtered_columns = [col for col in X.columns if not any(substring in col for substring in pfas_list)]

    # X = X[filtered_columns]
    
    ##############################################

    # y = y['PFOA__NVT__ug/l']
    
    epsilon = 1e-6
    # print(f'y before transformation: {y}, shape: {y.shape}')

    # log transform y 
    y_transformed = np.log(y + epsilon)

    # #X = X.drop('STN', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.25, random_state = 5)

    # print(f'Y train after transformation: {y_train}, shape: {y_train.shape}')
    
    
    #baseline_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test)
    
    
    ####################################################3
    

    # randomized_search(X_train, y_train)
    # # # Model training with cross validation 

    #RFR(X_train, y_train)
    #xgboost(X_train, y_train, X_test, y_test)
    
    # # # # Evaluate the model on training set
    # # # # and visualize the results
    #model_eval(X_test, y_test)

    # ################# FEATURE IMPORTANCE #######################

    #permutation(model, X_train, y_train)

main()


