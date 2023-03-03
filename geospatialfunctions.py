# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:50:33 2023

@author: Thiago Nascimento
Module with some usefull geospatial functions widely used for data-analysis.
"""

import geopandas as gpd
from shapely.geometry import Point
from plotly.offline import plot
import contextily as cx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
import datetime
import matplotlib.dates as mdates


warnings.simplefilter(action='ignore', category=Warning)


#%% 1. Define a function for plot:

# This function is used for generating a quick map plot with the desired points 
# and a background map in background.

def plotpoints(plotsome: pd.pandas.core.frame.DataFrame, showcodes = False, figsizeproj = (15, 30)):
    """
    Inputs
    ------------------
    plotsome: dataset[Index = Code; columns = [Longitude, Latitude]]: 
        dataframe with the codes as the index, and with at least two columns so-called
        "Longitude" and "Latitude" (in EPSG: 4326). 
    
    showcodes: 
        By default it is set as "False". If "True" it will show the codes from the index. 
    
    Returns
    --------------------
    plt.plot: The output is a plt.plot with the points spatially distributed in the area. 
    A background map can be also shown if your coordinate system "crsproj" is set to 'epsg:4326'.
        
    """    
    
    crs={'init':'epsg:4326'}
    geometry=[Point(xy) for xy in zip(plotsome["Longitude"], plotsome["Latitude"])]
    geodata=gpd.GeoDataFrame(plotsome,crs=crs, geometry=geometry)
    geodatacond = geodata

    # The conversiojn is needed due to the projection of the basemap:
    geodatacond = geodatacond.to_crs(epsg=3857)

    # Plot the figure and set size:
    fig, ax = plt.subplots(figsize = figsizeproj)

    #Organizing the legend:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    #Ploting:
    #geodatacond.plot(ax=ax, column='PercentageGaps', legend=True, cax = cax, cmap = "Reds")
    geodatacond.plot(ax=ax, cmap = "spring")

    if showcodes == True:
        geodatacond["Code"] = geodatacond.index
        geodatacond.plot(column = 'Code',ax=ax);
        for x, y, label in zip(geodatacond.geometry.x, geodatacond.geometry.y, geodatacond.index):
            ax.annotate(label, xy=(x, y), xytext=(1, 1), textcoords="offset points")
        plt.rcParams.update({'font.size': 12})
    
    else:
        pass
    
    
    
    cx.add_basemap(ax)
    
#%% #### 2. Define a function for plot several time-series in a single plot:

# This function is used for generating a quickly several subplots of different
# time series in a single plot. 

def plottimeseries(numr, numc, datatoplot: pd.pandas.core.frame.DataFrame, setylim = False, ymin = 0, ymax = 1, figsizeproj = (18, 11),
                   colorgraph = "blue", linewidthproj = 0.5, linestyleproj = "-",  ylabelplot = "P (mm)",
                   datestart = datetime.date(1981, 6, 1), dateend = datetime.date(2021, 12, 31),
                   setnumberofintervals = False, numberintervals = 2):
    
    """
    Inputs
    ------------------
    numr = Number of rows of your figure;
    numc = Numer of columns of your figure;
    datatoplot: dataframe[Index = Datetime; columns = [rain-gauges]]: 

    setylim = It is used when one needs to set a common y-lim for the graphs;
    ymin and ymax = only used when "setylim" is "True";
    figsizeproj = size of the generated figure in inches;
    colorgraph = linecolor of the graphs;
    linewidthproj = linewidth of the graphs;
    linestyleproj = linestyle of the graphs;
    ylabelplot = label of the time-series (assuming all time series are in the same units and type);
    
    datestart and dateend = datetime variable defining the time-interval of the plots;
    setnumberofintervals = It is used when one needs to set manually the number of intervals of 
    the x-axis in years;
    numberintervals = By default it is set to 2-years.
    
    Returns
    --------------------
    plt.plot: The output is a plt.plot with the graphs plot in subplots. 
        
    """   
    
    fig, axs = plt.subplots(int(numr),int(numc), figsize = figsizeproj)

    i = 0

    for col in datatoplot.columns:
    
        plot_data = datatoplot.loc[:,col].dropna()
    
        name = col
    
        num_rows = axs.shape[0]
        num_cols = axs.shape[1]

    
        colauxs = [i for i in range(num_cols) for _ in range(num_rows)] 
        rowauxs = [*range(num_rows)] * num_cols
    
        colaux, rowaux = colauxs[i], rowauxs[i]
    
        axs[rowaux,colaux].plot(plot_data.index.values, plot_data.values, linestyle = linestyleproj, label=col, linewidth = linewidthproj, markersize=2, color = colorgraph)
        axs[rowaux,colaux].set_title(name, loc='left')
        axs[rowaux,colaux].set_xlim([datestart, dateend])
        
        if setnumberofintervals == True:
            axs[rowaux,colaux].xaxis.set_major_locator(mdates.YearLocator(int(numberintervals)))
        
        if setylim == True:
            axs[rowaux,colaux].set_ylim(ymin, ymax)
        else:
            pass
        
        if colaux == 0:
            axs[rowaux,colaux].set_ylabel(ylabelplot)
    
        i = i + 1

    plt.rcParams.update({'font.size': 8})
    plt.tight_layout()
    
    return plt.show()
    
#%% 3. Define a function for plot several box-plots in a single plot:

# This function is useful for the plot of several boxplots from a big time-series
# Initiall this function is used in a dataframe of 1898 rain gauges being each labeled with an unique 
# index (Code) and categorized per Federation State (or Cluster). Moreover, the dataframe has a column of
# maximum precipitation per code, which will be used for the boxplots.
# Therefore, the boxplots will be plot per State (and not per Code). 
# For different analysis the code might as well need to be adapted.


def plotboxplots(numr, numc, datatoplot: pd.pandas.core.frame.DataFrame, setylim = False, 
                 ymin = 0, ymax = 1, figsizeproj = (18, 11), ylabelplot = "P (mm)"):
    
    """
    Inputs
    ------------------
    numr = Number of rows of your figure;
    numc = Numer of columns of your figure;
    datatoplot: dataframe[Index = Codes; columns = [Cluster, Statistical descriptor]]: 

    setylim = It is used when one needs to set a common y-lim for the graphs;
    ymin and ymax = only used when "setylim" is "True";
    figsizeproj = size of the generated figure in inches;
    ylabelplot = label of the time-series (assuming all time series are in the same units and type);
    
    Returns
    --------------------
    plt.plot: Boxplot. 
        
    """   
    fig, axs = plt.subplots(int(numr),int(numc), figsize = figsizeproj)

    i = 0
    
    for col in datatoplot.Cluster.unique():
    
        plot_data = datatoplot[datatoplot["Cluster"] == col].loc[:,"max"]
    
        name = col
        
        num_rows = axs.shape[0]
        num_cols = axs.shape[1]

        
        colauxs = [i for i in range(num_cols) for _ in range(num_rows)] 
        rowauxs = [*range(num_rows)] * num_cols
    
        colaux, rowaux = colauxs[i], rowauxs[i]
    
        axs[rowaux,colaux].boxplot(plot_data)
        axs[rowaux,colaux].set_title(name, loc='left')
        
        
        if setylim == True:
            axs[rowaux,colaux].set_ylim(ymin, ymax)
        else:
            pass
        
        if colaux == 0:
            axs[rowaux,colaux].set_ylabel(ylabelplot)
    
        i = i + 1

    plt.rcParams.update({'font.size': 8})
    plt.tight_layout()

    return plt.show()

#%% 4. Make a df.describe considering a cluster:

# This function is useful for the quick computation of the main statistical descriptors
# such as: min, max, median and percentils of an initial time-series per cluster. 
# For example, one may have a initial time-series of several rain-gauge considering 
# monthly precipitation data and information about potential clusters (or regions). then:
    # 1. This function compute the maximum, minimum, average or other descriptor for each rain-gauge;
    # 2. The statistical descriptors of this descriptor are computed per cluster (region). 

def describeclusters(dataset: pd.pandas.core.frame.DataFrame, clusters: pd.pandas.core.frame.DataFrame, 
                     statisticaldescriptor = "mean", clustercolumnname = "Cluster"): 
    
    """
    Inputs
    ------------------

    dataset: dataframe[Index = Datetime; columns = [rain-gauges]]
    clusters: dataframe[Index = Code just as the columns of dataset; columns = clusters: 
    statisticaldescriptor: {"mean", "count", "std", "min", "25%", "50%", "75%", "max"}                    
    clustercolumn: Column cluster's name in the cluster dataframe.
        
    # It is essential that the columns of the dataframe dataset are the same as the index in the dataframe clusters. 
    
    Returns
    --------------------
    stationsdescriptor: dataframe[Index = Rain-gauges; columns = [Clusters, statisticaldescriptor]]
    clustersdescribe: dataframe[Index = Clusters; columns = ["mean", "min", "P25", "950", "25%", "P75",
                                                             "P90", "P95", "P99", "max", "P25 + 1.5IQR"]] 
        
    """   
    fsummary = dataset.describe()
    stationsdescriptor = pd.DataFrame(index = clusters.index, columns= ["Cluster"], data = clusters.loc[:, clustercolumnname].values)
    stationsdescriptor[statisticaldescriptor] = fsummary.T[statisticaldescriptor].values
    
    clustersdescribe = stationsdescriptor.groupby(by=["Cluster"]).mean()
  
    clustersdescribe.rename(columns = {statisticaldescriptor:'mean'}, inplace = True)
    clustersdescribe["min"] = stationsdescriptor.groupby(by=["Cluster"]).min()
    clustersdescribe["P25"] = stationsdescriptor.groupby(by=["Cluster"]).quantile(q = 0.25)
    clustersdescribe["P50"] = stationsdescriptor.groupby(by=["Cluster"]).quantile(q = 0.5)
    clustersdescribe["P75"] = stationsdescriptor.groupby(by=["Cluster"]).quantile(q = 0.75)
    clustersdescribe["P90"] = stationsdescriptor.groupby(by=["Cluster"]).quantile(q = 0.90)
    clustersdescribe["P95"] = stationsdescriptor.groupby(by=["Cluster"]).quantile(q = 0.95)
    clustersdescribe["P99"] = stationsdescriptor.groupby(by=["Cluster"]).quantile(q = 0.99)
    clustersdescribe["max"] = stationsdescriptor.groupby(by=["Cluster"]).max()
    clustersdescribe["Q1+1.5IQR"] = clustersdescribe["P25"] + (clustersdescribe["P75"] - clustersdescribe["P25"])*1.5
    
    return stationsdescriptor, clustersdescribe

#%% 5. Generate a grid centroids table from initial Lat/Lon and spacing data:

# The main application is for generating centroids of sattelite precipitation grid data (e.g., TRMM).
# Observations:

# (a) The latitude and longitude are computed as being from left to right and from upper to down;
# (b) Pay attention on where you have the space positive or negative, for instance, for Paraiba and TRMM, 
# 0.25 is negative for latitude and positive for longitude. Try to pay attention on where is the (0, 0) of 
# the equator and Greenwich. 
# Lat_final and lon_final must be set with one extra, because python does not consider the last.

def generategridcentroids(lat_initial, lat_final, lon_initial, lon_final, lat_spacing, lon_spacing, crsproj = 'epsg:4326'):
    """
    Inputs
    ------------------

    lat_initial: Latitude value in the centroid located at the upper left (origin)
    lat_final: Latitude value in the centroid located at the lower right + lat_spacing *
    lon_initial: Longitude value in origin (upper left) 
    lon_final: Longitude value in the centroid located at the lower right + lon_spacing *
    lat_spacing: latitude resolution
    lon_spacing: longitude reolution
    crsproj: If you are using WGS84, you the code you plot for you a background map, if not, it will just plot the points.
    
    * Lat_final and lon_final must be set with one extra, because python does not consider the last.
    
    Returns
    --------------------
    coord_grids: dataframe[Index = Centroid IDs; columns = [Lat, Lon]]
    plt.plot: Scatter plot showing the generated grid for conference. 
        
    """       
    
    
    
    lat = np.arange(lat_initial, lat_final, lat_spacing)
    lon = np.arange(lon_initial, lon_final, lon_spacing)
    
    num_rows = len(lat) * len(lon)
    coord_grids = pd.DataFrame(np.nan, index = range(num_rows), columns = [['Lat', 'Lon']])
    
    z = 0
    for i in lat:
        j = 0
        for j in lon:
            coord_grids.iloc[z, 0]  = i
            coord_grids.iloc[z, 1] = j
            z = z + 1
    coord_grids = pd.DataFrame(coord_grids)
    
    if crsproj == 'epsg:4326':
        coords_df = pd.DataFrame({'GridID': range(len(coord_grids)),
                         'Lat': coord_grids.Lat.values[:,0],
                         'Lon': coord_grids.Lon.values[:,0]})
        
        crs = {'init': crsproj}
        
        geometry=[Point(xy) for xy in zip(coords_df["Lon"], coords_df["Lat"])]
        geodata=gpd.GeoDataFrame(coords_df,crs=crs, geometry=geometry)
        geodatacond = geodata
        # The conversiojn is needed due to the projection of the basemap:
        geodatacond = geodatacond.to_crs(epsg=3857)

        # Plot the figure and set size:
        fig, ax = plt.subplots()

        #Organizing the legend:
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.1)

        #Ploting:
        geodatacond.plot(ax=ax)
        cx.add_basemap(ax)
    else:
        plt.scatter(coord_grids.Lon, coord_grids.Lat)
        
    return coord_grids
