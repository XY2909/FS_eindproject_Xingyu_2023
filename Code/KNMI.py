import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import numpy as np


# df_meetlocaties = pd.read_csv(r'c:\Users\panxi\RP\Meetlocaties alle jaren\lew_meetpunten_2019.csv', on_bad_lines = 'skip',  sep = ";", low_memory = False)

# df2019 = pd.read_excel(r'c:\Users\panxi\RP\new_df2.xlsx')

# # print(df2019.shape)

# # list of sampling locations 2019
# locations = df2019['locations'].tolist()

# # print(len(set(locations)))
def remove_hyphens(date_string):
    return date_string.replace('-', '')
# indices = df_meetlocaties['Meetobject.code'].isin(locations)
def prepare():
    df = pd.read_excel(r"C:\Users\panxi\RP\final\test2.xlsx")


    # get unique locations 
    df['locations_f'] = df['Parameter__Hoedanigheid__Eenheid'].str.split("__").str[1]
    df['dates_f'] = df['Parameter__Hoedanigheid__Eenheid'].str.split("__").str[2]

    print(df.shape)

    # Identify duplicate strings in the 'locations' column
    duplicate_mask = df['locations_f'].duplicated(keep=False)

    # Filter the DataFrame to keep only non-duplicate rows
    df = df[~duplicate_mask]

    
    col_t0_keep = ['x_gps', 'y_gps', 'locations_f']
    df_small = df[col_t0_keep]

    df_small.to_excel(r"C:\Users\panxi\RP\final\loc_coord.xlsx")


    print(df_small.shape)
    print(df_small)

# prepare()

# This function returns distance of two points in km
def calculate_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    # convert coordinates to radians
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    # calculate the differences in coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # apply Haversine formula to calculate the distance
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance

def most_close_station():
    df_weerstation = pd.read_csv(r"C:\Users\panxi\RP\weerstation_loc.txt", sep = ',')

    df2 = pd.read_excel(r"C:\Users\panxi\RP\final\loc_coord.xlsx")

    ds = []
    ds1 = []

    STNs_to_avoid = [225, 248, 258, 285, 313, 324, 331, 343]

    for index, row in df2.iterrows():

        distances = []

        # get coordinates from sampling points
        lat1 = row['x_gps']
        lon1 = row['y_gps']

        # get coordinates from weather stations
        for index, row in df_weerstation.iterrows():

            lat2 = row['LAT(north)']
            lon2 = row['LON(east)']

            distance = calculate_distance(lat1, lon1, lat2, lon2)
            distances.append((distance, row['STN']))

        
        # calculate distance to every STN
        meest_dichtbij_tuple = min(distances, key=lambda x: x[0])
        meest_dichtbij_STN = meest_dichtbij_tuple[1]

       
        ds1.append(meest_dichtbij_STN)

        # filter STN -> choose next closest wheather station
        new_list = [x for x in distances if x!= meest_dichtbij_tuple]

        while meest_dichtbij_STN in STNs_to_avoid:

            meest_dichtbij_tuple = min(new_list, key=lambda x: x[0])
            meest_dichtbij_STN = meest_dichtbij_tuple[1]

            # update distances to STN
            new_list = [x for x in new_list if x!= meest_dichtbij_tuple]

        ds.append(meest_dichtbij_STN)
        distances.clear()
        new_list.clear()
        

    df2['meest dichtbij weerstation'] = ds1
    df2['meest dichtbij weerstation filtered'] = ds

    df2.to_excel(__path__)


def link_station():

    new_df = pd.read_excel(__path__)
    df_knmi = pd.read_excel(r"...\knmi_weatherdata.xlsx")
    df_stations = pd.read_excel(r'C:\Users\panxi\RP\final\loc_coord_stn.xlsx')

    complete_df = pd.DataFrame()

    locations = []
    dates = []
    stns = []

    daily_weathers = []
    removed_index = []

    count = 0
    for index, row in new_df.iterrows():

        # get the sampling date
        date = int(row['dates'])
        
        # get sampling location
        loc = row['locations']

        
        try:
            # get most close weather station number
            station_number = int(df_stations.loc[(df_stations['locations_f'] == loc, 'meest dichtbij weerstation filtered')].iloc[0])

            # stations with no data for 2019
            if station_number not in [210, 265, 311]:

                dates.append(date)
                stns.append(station_number)
                locations.append(loc)

                # filter the rows that match the values in col1 and col2
                daily_weather = df_knmi.loc[(df_knmi['STN'] == station_number) & (df_knmi['YYYYMMDD'] == date)]

                daily_weathers.append(daily_weather)

            else:
                removed_index.append(index)
                count += 1
            
        except IndexError:
            #print(f'There is no STN info available for sample location: {loc}, date: {date}')
            removed_index.append(index)
            

            # complete_df = complete_df.append(daily_weather, ignore_index=False)

        # print(len(locations))
        # print(len(stns))
        # print(len(dates))

    # merge dfs in a list to one df
    daily_weathers = pd.concat(daily_weathers, axis=0, ignore_index=True)

    # fill in the empty dataframe
    complete_df['locations'] = locations
    complete_df['dates'] = dates
    complete_df['stn'] = stns

    print('length removed index', len(removed_index))
    print('count:', count)

    # concatenate two dataframes side by side
    df_combined = pd.concat([complete_df, daily_weathers], axis=1)


    print('removed index: ', removed_index)

    df_combined.to_excel(__path__)


# link_station()


removed_index = [70, 97, 144, 150, 241, 281, 282, 285, 320, 400, 425, 494, 545, 568, 629, 657, 714, 747, 828, 840, 852, 917, 930, 999, 1024]


def combine():

    df_knmi = pd.read_excel(r'C:\Users\panxi\RP\knmi2019_STN_mod.xlsx')

    print('length df_knmi: ', len(df_knmi)) 

    df_chemical = pd.read_csv(r'c:\Users\panxi\RP\Deelprocessen python\df longs\df2019_long.csv', on_bad_lines = 'skip',  sep = ";", low_memory = False)

    
    print('length of df_chemical before removing: ', len(df_chemical))

    # drop sampling points without weather data
    df_chemical = df_chemical.drop(index = removed_index)

    print('length df_chemical after drop(): ', len(df_chemical))

    df_chemical = df_chemical.reset_index(drop=True)
    df_knmi = df_knmi.reset_index(drop=True)

    df_chem_knmi = pd.concat([df_chemical, df_knmi], axis=1)

    print('length combined df: ', len(df_chem_knmi))

    df_chem_knmi.to_excel(r'C:\Users\panxi\RP\chem_knmi_2019_STN_mod.xlsx')

#combine()

# These two functions convert wind direction in degrees to categorical values
def categorize_wind_direction(degrees: int) -> int:
        bins = [0, 23, 68, 113, 158, 203, 248, 293, 338, 360]
        labels_num = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        category = pd.cut([degrees], bins=bins, labels=labels_num, include_lowest=True)[0]
        return category

    
def change_ddvec():
    df = pd.read_excel(__path__)

    # define the bin edges and labels
    bins = [0, 23, 68, 113, 158, 203, 248, 293, 338, 360]
    labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S']
    labels_num = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    DDVEC_CAT = []

    # Replace string values with NaN and cast columns to float
    # replace empty strings or strings containing only spaces with NaN values
    df['DDVEC'] = df['DDVEC'].replace(r'^\s*$', np.nan, regex=True)
    df['DDVEC_cat'] = df['DDVEC'].apply(categorize_wind_direction)
    

    print(len(df['DDVEC']))
    print(df['DDVEC_cat'])

    # categorize the wind direction data
    #df['DDVEC_cat'] = pd.cut(df['DDVEC'], bins=bins, labels=labels_num, include_lowest=True)

    return df


# change_ddvec()
