import pandas as pd
import pyproj
from math import radians, sin, cos, sqrt, atan2
import numpy as np



def get_location():
    df_meetlocaties = pd.read_csv(r'c:\Users\panxi\RP\Meetlocaties alle jaren\lew_meetpunten_2019.csv', on_bad_lines = 'skip',  sep = ";", low_memory = False)

    df2019 = pd.read_excel(r'c:\Users\panxi\RP\new_df2.xlsx')

    new_df = pd.DataFrame()

    # list of sampling locations 2019
    locations = df2019['locations'].tolist()

    rd_xs = []
    rd_ys = []
    names = []
    dates = []
    
    print('unique locations 2019:', len(set(locations)))
    #print(len(df2019))

    # loc = df2019['Original'].apply(lambda x: x.split('__')[1])
    # date = df2019['Original'].apply(lambda x: x.split('__')[2])

    #  no_data = ['NL34_5012', 'NL34_6402']

    found = False
    for index, row in df2019.iterrows():

        loc = row['Original'].split("__")[1]
        date = row['Original'].split("__")[2]

        for meetloc in df_meetlocaties['Meetobject.code'].values:

            if meetloc == loc:
        
                index = df_meetlocaties[df_meetlocaties['Meetobject.code'] == meetloc].index[0]

                rd_x = df_meetlocaties.loc[index, 'GeometriePunt.X_RD']
                rd_y = df_meetlocaties.loc[index, 'GeometriePunt.Y_RD']

                rd_xs.append(rd_x)
                rd_ys.append(rd_y)
                names.append(loc)
                dates.append(date)
     
            
    new_df['rd_x'] = rd_xs
    new_df['rd_t'] = rd_ys
    new_df['names'] = names
    new_df['date'] = dates
    new_df['dates'] = new_df['date'].apply(remove_hyphens)

    # Find values in list2 but not in list1
    #diff2 = set(new_df['names'].to_list()) - set(locations)

def rd_to_gps():
    # Convert rd coordinates to gps coordinates 
  
    df2019 = pd.read_excel(__path__)

    rd_projection = pyproj.Proj('+init=EPSG:28992')  # Dutch RD projection
    wgs84_projection = pyproj.Proj('+init=EPSG:4326')  # WGS84 projection

    # Create a transformer to convert from RD to GPS
    transformer = pyproj.Transformer.from_proj(rd_projection, wgs84_projection)

    # Apply the transformer to x_rd and y_rd columns and assign to new columns
    df2019['x_gps'], df2019['y_gps'] = transformer.transform(df2019['rd_x'].values, df2019['rd_y'].values)


    df2019.to_excel(__path__)

# This function replace YYYY-MM-DD with YYYYMMDD
def remove_hyphens(date_string):
    return date_string.replace('-', '')



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
    # find the most close weather station 
    df_riool = pd.read_excel(r"...\rioolwaterzuiveringsinstallatie_locaties.xlsx")
    df2 = pd.read_excel(__path__)

    ds = []
    names = []

    for index, row in df2.iterrows():
        
        distances = []

        lat1 = row['x_gps']
        lon1 = row['y_gps']
        

        for index, row in df_riool.iterrows():

            lat2 = row['y_gps']
            lon2 = row['x_gps']

            distance = calculate_distance(lat1, lon1, lat2, lon2)
            
            RZIname = row['Name']

            distances.append((distance, RZIname))
        

        meest_dictbij, name = min(distances, key=lambda x: x[0])
        
        ds.append(meest_dictbij)
        names.append(name)

        distances.clear()

    df2['afstand tot meest dichtbij RWI'] = ds
    df2['naam RWI'] = names

    df2.to_excel(__path__)



# most_close_station()

