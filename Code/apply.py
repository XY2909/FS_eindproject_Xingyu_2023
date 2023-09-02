from collections import defaultdict
import itertools
from matplotlib.markers import MarkerStyle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import pyproj
from math import radians, sin, cos, sqrt, atan2
import numpy as np
from joblib import dump, load
import pyproj
import cloupy as cl
import cbsodata
import seaborn as sns
from scipy.stats import wilcoxon
import sklearn
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from joblib import Parallel, delayed
from scipy.stats import shapiro

from matplotlib import pyplot

def remove_hyphens(date_string):
    return date_string.replace('-', '')

def get_location():
    # P.S. rd correct format: divided by 10000 (remove 4 zeros)
    df_meetlocaties = pd.read_csv(__path__, on_bad_lines = 'skip',  sep = ";", low_memory = False)

    df2019 = pd.read_excel(__path__)

    print('shape df2019: ', df2019.shape)
    print('df2019 columns:', df2019.columns)
    print('number of unique columns:    ', len(set(df2019.columns)))

    columns_2019 = df2019.columns
    new_df = pd.DataFrame(columns=columns_2019)


    # df2019['locations'] = df2019['Parameter__Hoedanigheid__Eenheid'].apply(lambda x: x.split('__')[1])
    # df2019['date'] = df2019['Parameter__Hoedanigheid__Eenheid'].apply(lambda x: x.split('__')[2])

    rd_xs = []
    rd_ys = []
    names = []
    dates = []

    # loc = df2019['Original'].apply(lambda x: x.split('__')[1])
    # date = df2019['Original'].apply(lambda x: x.split('__')[2])

    #  no_data = ['NL34_5012', 'NL34_6402']

 
    for index_2019, row in df2019.iterrows():

        loc = row['Parameter__Hoedanigheid__Eenheid'].split("__")[1]
        date = row['Parameter__Hoedanigheid__Eenheid'].split("__")[2]

        for meetloc in df_meetlocaties['Meetobject.code'].values:

            if meetloc == loc:
                
                # Get index from corresponding row
                index = df_meetlocaties[df_meetlocaties['Meetobject.code'] == meetloc].index[0]

                # Get coordinates
                rd_x = df_meetlocaties.loc[index, 'GeometriePunt.X_RD']
                rd_y = df_meetlocaties.loc[index, 'GeometriePunt.Y_RD']
                
                # Remove thousands separator and convert to integer
                rd_x = int(rd_x.replace('.', ''))
                rd_y = int(rd_y.replace('.', ''))

                rd_xs.append(rd_x)
                rd_ys.append(rd_y)
                # names.append(loc)
                # dates.append(date)
            
                # copy entire row from df2019 and append location data
                # store those in new_df
                new_df.loc[index_2019] = df2019.loc[index_2019]
                

                break
        
            
            
    new_df['rd_x'] = rd_xs
    new_df['rd_y'] = rd_ys

    print('shape new df: ', new_df.shape)

    # new_df['rd_x'] = rd_xs
    # new_df['rd_y'] = rd_ys

    # new_df['names'] = names
    # new_df['date'] = dates
    # new_df['dates'] = new_df['date'].apply(remove_hyphens)
    # new_df['Parameter__Hoedanigheid__Eenheid'] = df2019['Parameter__Hoedanigheid__Eenheid']

    rd_projection = pyproj.Proj('+init=EPSG:28992')  # Dutch RD projection
    wgs84_projection = pyproj.Proj('+init=EPSG:4326')  # WGS84 projection

    # Create a transformer to convert from RD to GPS
    transformer = pyproj.Transformer.from_proj(rd_projection, wgs84_projection)

    # Apply the transformer to x_rd and y_rd columns and assign to new columns
    df2019['x_gps'], df2019['y_gps'] = transformer.transform(new_df['rd_x'].values , new_df['rd_y'].values )


    # # Find values in list2 but not in list1
    # #diff2 = set(new_df['names'].to_list()) - set(locations)

    # # df['dates'] = df['dates'].apply(remove_hyphens)
    new_df.to_excel(r'c:\Users\panxi\RP\final\test.xlsx')

def back_transform(scaled_number):

    epsilon = 1e-6

    number = np.exp(scaled_number) - epsilon

    print(f'after back transformation: {number}')

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

def rd_to_gps():

    df2019 = pd.read_excel(r'c:\Users\panxi\RP\test\df2019_noPFOA_filtered.xlsx')

    rd_projection = pyproj.Proj('+init=EPSG:28992')  # Dutch RD projection
    wgs84_projection = pyproj.Proj('+init=EPSG:4326')  # WGS84 projection

    # Create a transformer to convert from RD to GPS
    transformer = pyproj.Transformer.from_proj(rd_projection, wgs84_projection)

    # Apply the transformer to x_rd and y_rd columns and assign to new columns
    df2019['x_gps'], df2019['y_gps'] = transformer.transform(df2019['rd_x'].values, df2019['rd_y'].values)


    df2019.to_excel(r'c:\Users\panxi\RP\test\df2019_noPFOA_filtered.xlsx')


# draw interpolation map 
def interpolate(df):

    df_total = df 
    #print('df_total:  ', df_total)


    # slice of dataframe containing actual measured pfoa concentrations in jan 2019
    df_actual_measured = df[:33]

    print('df_actual_measured:  ', df_actual_measured)



    which_df = [df_actual_measured, df_total]

    figure_names = ['measured', 'measured and predicted']
    
    for i in range(2):

        imap = cl.m_MapInterpolation(
        shapefile_path = r'c:\Users\panxi\RP\extra data\gadmin\gadm41_NLD_0.shp', # main shape: country boundary
        crs='epsg:4326', # specify the shapefile coordinates system
        dataframe = which_df[i] # pass manually the dataframe
        )

        imap.draw(

            zoom_in=[(3.0, 7.4), (50.6, 53.7)],
            levels=np.arange(-7.6, -4.50, 0.1), # range of data
            add_shape={

                # the shapefiles were downloaded from the GADM website under GADM license.
                # second layer: provincial boundaries 
                r'c:\Users\panxi\RP\extra data\gadmin\gadm41_NLD_1.shp': 'crs=epsg:4326, linewidth=0.4, linestyle=dotted, color=black',
                
                # water lines, obtained from Tessa
                r'c:\Users\panxi\RP\extra data\water\NLD_water_lines_dcw.shp': 'crs=epsg:4326, linewidth=0.4, linestyle=solid, color=blue'},  
                

            cmap='Blues',
            
            interpolation_method='linear',
            interpolation_within_levels=True,

            boundaries_lw=0.3,

            show_points=True,
            
            show_grid=True,

            #show_contours=True,
            #contours_levels=np.arange(-7.6, -4.50, 0.1),

            xticks=np.arange(3, 7.5, 1),
            # yticks=[10, 12, 0, -1, -2, -3, -4, -23, -27],
            figsize=(4, 4.7),
            
            cbar_title = 'PFOA concentrations [log(ug/l)]',
            title = f'Fig. {i+1}. Spatial layout of the {figure_names[i]} PFOA \nconcentrations in Netherlands in Jan 2019',
            title_ha = 'center',
            title_x_position=0.5,


            save = rf'C:\Users\panxi\RP\final\interpolation_{figure_names[i]}PFOA.png'
        )





################################# APPLY MODEL #################################################

# df = pd.read_excel(__path__)

# ignore the first two columns
# df2 = df.iloc[:, 5:]


# df2 = df2.replace(',', '.', regex=True)


# # # check if df contains any nan values
# # # print(df2.isna().any().any())

# rf = load(r'c:\Users\panxi\RP\final\top5.joblib')

# # # # Same unit conversion


# log_predicted_pfoa = rf.predict(df2)

# # # # Reverse the log transformation
# original_values_back = np.exp(log_predicted_pfoa)


# df['Predicted PFOA concentrations'] = original_values_back
# df['log predicted pfoa'] = log_predicted_pfoa


########################## START INTERPOLATION #########################
df = pd.read_excel(__path__)


# df_actual = df[:34]

# print(df_actual)
# print('number of unique locations: ', len(df_actual['x_gps'].unique()) )


# # print(f"min log pfoa: {min(df['Predicted log PFOA concentrations'])}, max: {max(df['Predicted log PFOA concentrations'])}")
#df = df.sort_values(by ='dates')

# # print(len(df))

# x : latitude
# y : longtitude
# the order does matter!!
df = df[['log predicted pfoa', 'y_gps', 'x_gps']]

selected_rows = df[df['log predicted pfoa'] > -4.61]
print(selected_rows.shape)

imap = cl.m_MapInterpolation(
        shapefile_path = r'...\gadm41_NLD_0.shp', # main shape: country boundary
        crs='epsg:4326', # specify the shapefile coordinates system
        dataframe = selected_rows # pass manually the dataframe
        )

imap.draw(

    zoom_in=[(3.0, 7.4), (50.6, 53.7)],
    levels=np.arange(-4.61, -4.25, 0.1), # range of data
    add_shape={

                # the shapefiles were downloaded from the GADM website under GADM license.
                # second layer: provincial boundaries 
                r'c:\Users\panxi\RP\extra data\gadmin\gadm41_NLD_1.shp': 'crs=epsg:4326, linewidth=0.4, linestyle=dotted, color=black',
                
                # water lines, obtained from Tessa
                r'c:\Users\panxi\RP\extra data\water\NLD_water_lines_dcw.shp': 'crs=epsg:4326, linewidth=0.4, linestyle=solid, color=blue'},  
                

            cmap='Reds',
        
            #interpolation_method= 'linear',
            #interpolation_within_levels=True,

            boundaries_lw=0.3,

            show_points=True,
            
            show_grid=True,

            #show_contours=True,
            #contours_levels=np.arange(-7.6, -4.50, 0.1),

            xticks=np.arange(3, 7.5, 1),
            # yticks=[10, 12, 0, -1, -2, -3, -4, -23, -27],
            figsize=(4, 4.7),
            
            cbar_title = 'PFOA concentrations [log(ug/l)]',

            save = __path__
        )



########### merge the individual interpolation maps side-by-side ##########

# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)



# #fig.suptitle("Figure Title", y=0.2)


# ax1.axis('off')
# ax2.axis('off')

# img1 = plt.imread(__path__)
# img2 = plt.imread(__path__)

# ax1.imshow(img1)
# ax2.imshow(img2)

# ax1.set_aspect(1)

# print(img1.shape[1]) #1294
# print(img1.shape[0]) #1249

# print(img2.shape[1]) #1477
# print(img2.shape[0]) #1249

# ax2.set_aspect(1)

# plt.subplots_adjust(wspace= -0.05)

# plt.savefig(__path__, 
#                 dpi=300, bbox_inches='tight', 
#                 pad_inches=0.05, facecolor='white'
#                 )


############# WILCOXON TEST ##############################

# Assuming you have arrays of MSE values for each model
# mse_random_forest = np.array([0.21812935, 0.05690125, 0.04308791, 0.22942085, 0.21565443, 0.06047018, 0.03268835, 0.03163217, 0.05860009, 0.02805511])
# mse_xgboost = np.array([0.20097436, 0.07302496, 0.12390393, 0.29101611, 0.1992464, 0.06915583, 0.09177566, 0.03676157, 0.08311859, 0.06817482])


# # One-sided Wilcoxon test (greater)
# statistic, p_value_one_sided = wilcoxon(mse_xgboost, mse_random_forest, alternative='less')
# print("One-sided test:")
# print("Statistic:", statistic)
# print("p-value:", p_value_one_sided)

# print(f'mean of f5 is {np.mean(f5)}, mean of f8 is {np.mean(f8)}')


def plot_feature_selection():
    d_impurity = {'3':[0.12371701740803129, 0.06444548791120033],'4':[0.12709602841656503, 0.06884848497587667], '5':[0.11965602563613338, 0.06974648756669481], '6':[0.11638859834212696, 0.07197421794264666], '7':[0.1066718490167812, 0.07232528012897563], '8': [0.1014583243479295, 0.0709511580991106], '9': [0.09759877908685115, 0.06761795190439018], '10': [0.0944646166305996, 0.06829070010757804], '11': [0.0988885703608805, 0.07125377349206624], '12': [0.0997335644115139, 0.07275797955606962], '13': [0.09843433940893062, 0.07085326568929301], '14': [0.09691560522441005, 0.07050883000462456], '15': [0.09626642232010223, 0.07083298353990157], '16': [0.09572585208608256, 0.07081819488941142], '17': [0.0960877177322779, 0.07376192080567584], '18': [0.09880222773750627, 0.07067606418445836], '19': [0.093537983010165, 0.06983916214195945], '20': [0.093323394538287, 0.06990519932644851]}

    d_permut = {'3': [0.15559416884214397, 0.10333143167166817], '4': [0.1369546556170722, 0.0749982350813476], '5': [0.12779540435611736, 0.08109339280143214], '6': [0.122999419111457, 0.0788266474182503], '7': [0.10483721955458533, 0.0773739530133438], '8': [0.09809243249108418, 0.07781481312999311], '9': [0.10178918558093879, 0.07137733669297006], '10': [0.10118094936967548, 0.07190831806181323], '11': [0.10144620859245057, 0.0763888949798899], '12': [0.09832157217500967, 0.07384964822150826], '13': [0.09845744815448489, 0.07601786299052912], '14': [0.09630772517957073, 0.07154925982920671], '15': [0.09602730133698428, 0.069007682200934], '16': [0.09422353365411698, 0.06621007654615942], '17': [0.09589422832147575, 0.058098380741939716], '18': [0.09449840167315747, 0.06903697348895564], '19': [0.09153800587549112, 0.06662232506940867], '20': [0.09377401848047887, 0.06525300471619012]}

    numbers = list(d_impurity.keys())
    mean_values = [value[0] for value in d_impurity.values()]
    std_values = [value[1] for value in d_impurity.values()]


    key_perm = list(d_permut.keys())
    mean_values_perm = [value[0] for value in d_permut.values()]
    std_values_perm = [value[1] for value in d_permut.values()]



    # Convert std_values and std_values_perm to float
    std_values = np.array(std_values, dtype=float)
    std_values_perm = np.array(std_values_perm, dtype=float)
    std_values_comb= np.array(std_values_comb, dtype=float)

    # Calculate upper and lower bounds for confidence intervals
    upper_bound = mean_values + 1.96 * std_values / np.sqrt(len(mean_values))
    lower_bound = mean_values - 1.96 * std_values / np.sqrt(len(mean_values))

    # Plot the line with confidence intervals and create a legend handle
    line1, = plt.plot(numbers, mean_values, linestyle='-', label='MDI based')
    legend_handle1 = plt.fill_between(numbers, upper_bound, lower_bound, alpha=0.3)

    # Repeat the same process for the second line plot
    upper_bound_perm = mean_values_perm + 1.96 * std_values_perm / np.sqrt(len(mean_values_perm))
    lower_bound_perm = mean_values_perm - 1.96 * std_values_perm / np.sqrt(len(mean_values_perm))
    line2, = plt.plot(key_perm, mean_values_perm, linestyle='--', label='Permutation based')
    legend_handle2 = plt.fill_between(key_perm, upper_bound_perm, lower_bound_perm, alpha=0.3)
    
    # Combine the legend handles and labels
    legend_handles = [legend_handle1, legend_handle2]
    legend_labels = ['MDI based (95% CI)', 'Permutation based (95% CI)']

    # Set labels and title
    plt.xlabel('Number of selected features after ranking')
    plt.ylabel('Average MSE across 10 folds')
    plt.title('Feature reduction')
    plt.legend(legend_handles, legend_labels)
    plt.ylim(0, 0.25)
    plt.savefig(__path__)


#plot_feature_selection()

# plt.plot(numbers, mean_values, linestyle='-')

# plt.plot(key_perm, mean_values_perm, linestyle='-')

# # Plot mean values with error bars
# plt.errorbar(numbers, mean_values, yerr=std_values, capsize=4, label='MDI based')
# plt.errorbar(key_perm, mean_values_perm, yerr=std_values_perm, capsize=4, label='Permutation based')



# # Set labels and title
# plt.xlabel('Number of features')
# plt.ylabel('Average MSE across 10 folds')
# plt.title('Feature reduction')
# plt.legend()
# plt.ylim(0, 0.30)

# Display the plot

# plt.savefig(__path__)
# plt.show()



######### GET COMBINED RANK ############
# top_20_permut = ['T__NVT__oC', 'TX', 'Cd__NVT__ug/l', 'Cu__NVT__ug/l', 'Corg__NVT__ug/l',
#        'Durn__NVT__ug/l', 'V__NVT__ug/l', 'Sb__nf__ug/l', 'O2__NVT__%',
#        'Cl__NVT__ug/l', 'Clidzn__NVT__ug/l', 'NO2__Nnf__ug/l',
#        'TC4ySn__NVT__ug/l', 'Dist_to_RWI', 'Sb__NVT__ug/l', 'Corg__Cnf__ug/l',
#        'Ti__nf__ug/l', 'DmtnmdP__NVT__ug/l', 'B__NVT__ug/l', 'Ni__nf__ug/l']


# top_20_permut = top_20_permut[::-1]

# top_20_mdi = ['Ti__nf__ug/l', "Ni__nf__ug/l", 'Cl__NVT__ug/l', 'B__NVT__ug/l', 'Corg__Cnf__ug/l', 'Rb__nf__ug/l',    
# 'Dist_to_RWI',  'Sb__nf__ug/l', 'Durn__NVT__ug/l',  'Sb__NVT__ug/l ', 'DmtnmdP__NVT__ug/l','PCB180__NVT__ug/l',  'UN', 'DR', 'NO2__Nnf__ug/l', 'Clidzn__NVT__ug/l', 'TC4ySn__NVT__ug/l','V__NVT__ug/l','T4ClC2e__NVT__ug/l','Corg__NVT__ug/l']




# all_features = set(top_20_permut + top_20_mdi)
# combined_ranks = {}

# # Initialize ranks
# for feature in all_features:
#     combined_ranks[feature] = float('inf')  # Assign a large initial rank

# # Update ranks based on list1
# for i, feature in enumerate(top_20_permut):
#     combined_ranks[feature] = min(combined_ranks[feature], i + 1)

# # Update ranks based on list2
# for i, feature in enumerate(top_20_mdi):
#     combined_ranks[feature] = min(combined_ranks[feature], i + 1)


# # sort ranks
# sorted_dict = dict(sorted(combined_ranks.items(), key=lambda x: x[1]))

# print(sorted_dict)
# print(list(sorted_dict)[:20])

# print(len(sorted_dict))



# Combined_rank = ['Ni__nf__ug/l', 'Ti__nf__ug/l', 'B__NVT__ug/l', 'Cl__NVT__ug/l', 'DmtnmdP__NVT__ug/l', 'Corg__Cnf__ug/l', 'Sb__NVT__ug/l', 'Rb__nf__ug/l', 'Dist_to_RWI', 'Sb__nf__ug/l', 'TC4ySn__NVT__ug/l', 'NO2__Nnf__ug/l', 'Durn__NVT__ug/l', 'Clidzn__NVT__ug/l', 'Sb__NVT__ug/l', 'PCB180__NVT__ug/l', 'O2__NVT__%', 'UN', 'V__NVT__ug/l', 'DR']


def do_thing(number_of_cols):

    best_cols = defaultdict(lambda: [])
    for column in col_names:
        best_cols[df[column].notna().sum()].append({column})

    # for filled in sorted(list(best_cols.keys()), reverse=True):
    #     print(filled, best_cols[filled])

    while True:
        while not best_cols[sorted(list(best_cols.keys()), reverse=True)[0]]:
            best_cols.pop(sorted(list(best_cols.keys()), reverse=True)[0])

        best = list(best_cols[sorted(list(best_cols.keys()), reverse=True)[0]].pop())

        for column in col_names:
            if column in best:
                continue

            score = df[[column] + best].notna().all(axis=1).sum() # metric is total overlap between columns

            best_cols[score].append(set([column] + best))
            if len([column] + best) == number_of_cols:
                print(best_cols)
                print(score, [column] + best)
                return
            
    print(best_cols[sorted(list(best_cols.keys()), reverse=True)[0]])

# do_thing(8)



############### PEARSONS CORRELATION COEFFICIENT ##############
# selected_features = []

# positive_features = []
# negative_features = []

# # Iterate over each feature in your DataFrame
# for feature in df.columns:
#     if feature != target_variable:
#         # Calculate Pearson correlation coefficient and p-value
#         correlation_coef, p_value = pearsonr(df[feature], df[target_variable])
        
#         # Check if the p-value is below 0.05
#         if p_value < 0.05:
#             selected_features.append(feature)
        
#         # Store the feature in the respective list based on its sign of correlation coefficient
#         if correlation_coef > 0:
#             positive_features.append((feature, correlation_coef))
#         else:
#             negative_features.append((feature, correlation_coef))

# # Sort the positive features list based on the correlation coefficient in descending order
# sorted_positive_features = sorted(positive_features, key=lambda x: x[1], reverse=True)[:10]

# # Sort the negative features list based on the correlation coefficient in ascending order
# sorted_negative_features = sorted(negative_features, key=lambda x: x[1])[:10]

# # Print the top 10 features with highest positive correlation
# print("Top 10 features with highest positive correlation:")
# for feature, correlation_coef in sorted_positive_features:
#     print(f"Feature: {feature}\tCorrelation Coefficient: {correlation_coef}")

# # Print the top 10 features with highest negative correlation
# print("\nTop 10 features with highest negative correlation:")
# for feature, correlation_coef in sorted_negative_features:
#     print(f"Feature: {feature}\tCorrelation Coefficient: {correlation_coef}")



# # Convert the DataFrame to a numpy array or pandas DataFrame
# X = df.values  # If using numpy array
# # X = df  # If using pandas DataFrame

# y = df[target_variable]  # Specify the target variable

# # Compute mutual information between each feature and the target variable
# mutual_info = mutual_info_regression(X, y)

# # Create a dictionary to store feature names and their corresponding mutual information values
# feature_mi_dict = dict(zip(df.columns, mutual_info))

# # Sort the features based on their mutual information values in descending order
# sorted_features = sorted(feature_mi_dict, key=feature_mi_dict.get, reverse=True)

# # Print the feature names and their mutual information values
# for feature in sorted_features:
#     print(f"{feature}: {feature_mi_dict[feature]}")

# feature selection
def select_features(X_train, y_train, X_test):
 # configure to select all features
 # to select all features: k = 'all'
 fs = SelectKBest(score_func=mutual_info_regression, k='all')
 # learn relationship from training data
 fs.fit(X_train, y_train)
 # transform train input data
 X_train_fs = fs.transform(X_train)
 # transform test input data
 X_test_fs = fs.transform(X_test)

 return X_train_fs, X_test_fs, fs




############ Feature selection based on mutual info regression ###############
# split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=41)
# # feature selection
# X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)

# # what are scores for the features
# for i in range(len(fs.scores_)):
#  print('Feature %d: %f' % (i, fs.scores_[i]))
# # plot the scores
# pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
# pyplot.show()


# # define feature selection, select top 10
# fs = SelectKBest(score_func=f_regression, k=10)

# # apply feature selection
# X_selected = fs.fit_transform(X, y)


# # Get the indices of the selected features
# selected_feature_indices = fs.get_support(indices=True)

# # Get the names of the selected features
# selected_feature_names = X.columns[selected_feature_indices]

# # Print the names of the selected features
# print(selected_feature_names)

#######################
# y = df['PFOA__NVT__ug/l']
# X = df.drop(columns='PFOA__NVT__ug/l')


# hyperparameters_rfr = {'n_estimators': 900, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 10}
# Hyperparameters_xgb =  {'subsample': 0.8, 'n_estimators': 900, 'max_depth': 5, 'learning_rate': 0.01, 'colsample_bytree': 0.9}

# # Set up the models
# random_forest = RandomForestRegressor(**hyperparameters_rfr)
# xgboost = XGBRegressor(**Hyperparameters_xgb)

# # Set parameters for bootstrap resampling
# n_bootstrap = 100  # Number of bootstrap samples
# n_folds = 10       # Number of folds for cross-validation

# # Initialize arrays to store MSE for each model
# mse_random_forest = np.zeros(n_bootstrap)
# mse_xgboost = np.zeros(n_bootstrap)


# # Perform bootstrap resampling and cross-validation in parallel
# def perform_bootstrap(i):
#     # Generate bootstrap sample indices
#     indices = np.random.choice(len(X), size=len(X), replace=True)
#     X_bootstrap = X.iloc[indices]
#     y_bootstrap = y.iloc[indices]

#     # Perform cross-validation on the bootstrap sample
#     kf = KFold(n_splits=n_folds)
#     mse_fold_rf = np.zeros(n_folds)
#     mse_fold_xgb = np.zeros(n_folds)
#     fold = 0

#     for train_index, val_index in kf.split(X_bootstrap):
#         X_train, X_val = X_bootstrap.iloc[train_index], X_bootstrap.iloc[val_index]
#         y_train, y_val = y_bootstrap.iloc[train_index], y_bootstrap.iloc[val_index]

#         # Fit and evaluate Random Forest
#         random_forest.fit(X_train, y_train)
#         y_pred_rf = random_forest.predict(X_val)
#         mse_fold_rf[fold] = mean_squared_error(y_val, y_pred_rf)

#         # Fit and evaluate XGBoost
#         xgboost.fit(X_train, y_train)
#         y_pred_xgb = xgboost.predict(X_val)
#         mse_fold_xgb[fold] = mean_squared_error(y_val, y_pred_xgb)

#         fold += 1

#     print('mse_fold_rf: ', mse_fold_rf)
#     print('mse_fold_xgb: ', mse_fold_xgb)
#     return np.mean(mse_fold_rf), np.mean(mse_fold_xgb)

# results = Parallel(n_jobs=-1)(delayed(perform_bootstrap)(i) for i in range(n_bootstrap))

# # Extract the MSE results for each model from the results list
# mse_random_forest = np.array([result[0] for result in results])
# mse_xgboost = np.array([result[1] for result in results])


# print(mse_random_forest)
# print(mse_xgboost)
# # Perform statistical comparison (e.g., hypothesis testing, confidence intervals)
# # between the distributions of mse_random_forest and mse_xgboost as desired
# # (e.g., using t-tests, bootstrap tests, or confidence intervals).

# # Example: Calculate p-value using two-sample t-test
# from scipy.stats import ttest_ind
# t_stat, p_value = ttest_ind(mse_random_forest, mse_xgboost, equal_var=False)
# print("p-value:", p_value)


# color_map = {
#     'PFHpA__NVT__ug/l': 'darkblue',
#     'sverttPFOS__NVT__ug/l': 'darkblue',
#     'PFC5asfzr__NVT__ug/l': 'darkblue',
#     'PFNA__NVT__ug/l': 'darkblue',
#     'L_PFHpS__NVT__ug/l': 'darkblue',
#     'PFHxA__NVT__ug/l': 'darkblue',
#     'PFBA__NVT__ug/l': 'darkblue', 
#     'sverttPFHxS__NVT__ug/l': 'darkblue',
#     'PFPA__NVT__ug/l': 'darkblue', 
#     'L_PFHxS__NVT__ug/l': 'darkblue',
#     'PFDA__NVT__ug/l': 'darkblue',
#     'L_PFHxS__NVT__ug/l': 'darkblue',
#     'PFDA__NVT__ug/l': 'darkblue',
#     'FRD-903__NVT__ug/l':'darkblue'
# }


# # Plotting
# fig, ax = plt.subplots(figsize=(8, 10))
# plt.subplots_adjust(left=0.25)  # Adjust the left margin
# colors = ['darkblue' if name in color_map else 'steelblue' for name in feature_names]  # Assign colors based on condition

# y_pos = range(len(feature_names))
# ax.barh(y_pos, importance_values, align='center', color=colors)
# ax.set_yticks(y_pos)
# ax.set_yticklabels(feature_names)
# ax.invert_yaxis()
# ax.set_xlabel('Importances')
# ax.set_title('Top 20 Most Important Features on Full Dataset')

# plt.savefig(rf'C:\Users\panxi\RP\Results\final_eval(xgboost)\feature_importance_fulldata.png')





# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

# #fig.suptitle("Figure Title", y=0.2)

# ax1.axis('off')
# ax2.axis('off')


# img1 = plt.imread(rf'D:\Users\panxi\Pictures\feature_importance_fulldata.png')
# img2 = plt.imread(r'C:\Users\panxi\RP\Results\XGboost\exp3.png')

# ax1.imshow(img1)
# ax2.imshow(img2)


# ax1.set_aspect(1)
# ax2.set_aspect(1)

# # Add letter labels to the subplots
# ax1.text(0.03, 0.95, 'A', transform=ax1.transAxes, fontsize=8, fontweight='bold', va='top', ha='left')
# ax2.text(0.05, 0.95, 'B', transform=ax2.transAxes, fontsize=8, fontweight='bold', va='top', ha='left')

# ax1.set_aspect(1)

# print(img1.shape[1]) #1294
# print(img1.shape[0]) #1249

# print(img2.shape[1]) #1477
# print(img2.shape[0]) #1249

# ax2.set_aspect(1)


# plt.subplots_adjust(wspace= -0.08)

# plt.savefig(r'C:\Users\panxi\RP\Results\final_eval(xgboost)\combine.png', 
#                 dpi=300, bbox_inches='tight', 
#                 pad_inches=0.05, facecolor='white'
# #                 )


##################### SHAP ########################################
# df_long_complete = pd.read_excel(r'c:\Users\panxi\RP\chem_knmi_2019_STN_mod_remN.xlsx')

# df_long_complete = df_long_complete.replace(',', '.', regex=True)
# df_long_complete = df_long_complete.replace(r'^\s*$', np.nan, regex=True)

# df_long_complete = df_long_complete.iloc[:, 5:] 

# df_long_complete = df_long_complete.replace(',', '.', regex=True)

# # Replace string values with NaN and cast columns to float
# df_long_complete = df_long_complete.replace('', float('nan')).astype(float)


# y = df_long_complete['PFOA__NVT__ug/l']

# X = df_long_complete.drop('PFOA__NVT__ug/l', axis=1)
# X = X.drop('log pfoa', axis=1)
# X = X.drop('STN', axis=1)



# top5 = ['Ni__nf__ug/l', 'Dist_to_RWI', 'O2__NVT__%', 'UN', 'DR']

# X = X[top5]


# epsilon = 1e-6
    

# y_transformed = np.log(y + epsilon)

# X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.25, random_state = 42)


# model = load(rf'C:\Users\panxi\RP\final\top5.joblib')


# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)


# #shap.dependence_plot("PFHpA__NVT__ug/l", shap_values, X_test, interaction_index="sverttPFOS__NVT__ug/l")

# fig = shap.summary_plot(shap_values, X, show=False)
# plt.savefig(rf'C:\Users\panxi\RP\final\shap_top5.png')
# # results = model.evals_result()
# # print(results)
# # # plot learning curves
# # pyplot.plot(results['validation_0']['rmse'], label='train')
# # pyplot.plot(results['validation_1']['rmse'], label='test')
# # # show the legend
# # pyplot.legend()
# # # show the plot
# # pyplot.show()



# import matplotlib.pyplot as plt

# x = np.arange(3, 8)
# y1 = [492, 492, 445, 377, 89, 41]

# y2 = []
# # Create a figure and two subplots
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()  # Create a twin axes sharing the x-axis with ax1

# # Plot data on the first y-axis (ax1)
# ax1.plot(x, y1, 'b-', label='Y1 Data')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y1', color='b')
# ax1.tick_params(axis='y', colors='b')

# # Plot data on the second y-axis (ax2)
# ax2.plot(x, y2, 'r-', label='Y2 Data')
# ax2.set_ylabel('Y2', color='r')
# ax2.tick_params(axis='y', colors='r')

# # Set additional plot configurations
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')
# plt.title('Dual Y-Axis Plot')

# # Display the plot
# plt.show()



######### AZI & RWZ installaties distribution ##########
# shapefile = gpd.read_file(rf'C:\Users\panxi\RP\extra data\gadmin\gadm41_NLD_0.shp')


# shapfile2 = gpd.read_file(rf'C:\Users\panxi\RP\extra data\water\NLD_water_lines_dcw.shp')


# data = pd.read_excel(rf'C:\Users\panxi\RP\extra data\industrieel afvalwaterzuivering_locaties.xlsx')

# data_2 = pd.read_excel(rf'C:\Users\panxi\RP\extra data\rioolwaterzuiveringsinstallatie_locaties.xlsx')


# fig, ax = plt.subplots()

# # Plot the first shapefile in gray
# shapefile.plot(ax=ax, linewidth=0.5)

# shapfile2.plot(ax=ax, color='darkblue')

# # # Plot the second shapefile in green
# # shapefile2.plot(ax=ax, color='blue')

# # Plot the additional data points in blue
# # Plot the first set of data points in blue
# ax.scatter(data['x'], data['y'], color='black', label='IAZI', s=5)
# ax.scatter(data_2['x'], data_2['y'], color='red', label='RWZI', s=5)

# plt.show()

######## stats predicted PFOA ######### 


# df = pd.read_excel(r'C:\Users\panxi\RP\final\top5.xlsx')

# # Plot the distribution of the column
# sns.distplot(df['log predicted pfoa'], label='predicted pfoa')




# # Add labels and title
# plt.xlabel('Count')
# plt.ylabel('Frequency')
# plt.title('Distribution of predicted PFOA concentrations')

# # Display the plot
# plt.show()

#########
