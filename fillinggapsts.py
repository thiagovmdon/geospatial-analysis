# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:51:03 2022

@author: Thiago Nascimento
Code developed for filling missing data in time-series using multiple linear regression (MLR)
In this code we take into consideration the t-statistic value >= 2
"""

import pandas as pd
import numpy as np
import numpy
from scipy.spatial.distance import  cdist
from sklearn import linear_model
import tqdm as tqdm

import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
import contextily as cx

#%%
def summarygaps(df: pd.pandas.core.frame.DataFrame, coordsdf: pd.pandas.core.frame.DataFrame):
    """
    Inputs
    ------------------
    df: dataset[n x y]: 
        dataframe with already set an datetime index, and unique codes as columns. 
    
    coordsdf: dataset[y x 2]: 
        dataframe with its index as the same codes as the df columns, plus a X and Y gepgraphic coordinates (please follow this order). 
    
    Returns
    --------------------
    pandas.DataFrame [n x 4] with columns:
        'CoordX': Coordinates X
        
        'CoordY': Coordinates Y
        
        'NumGaps': Number of gaps per column
        
        'PercentageGaps': Percentage of gaps per column (%)
    """
    
    
    # Dealing with the data dataframe:
    df.index.name = 'dates'
    
    # Dealing with the coords dataframe:   
    coordsdf.index.name = 'Code'
    
    numrows= df.shape[0] #Total time lenght 

    # Calculate the percentage of failures per point:
    desc = df.describe()
    percerrorsdf = pd.DataFrame(index = coordsdf.index)
    
    percerrorsdf["CoordX"] = coordsdf.iloc[:,0]
    percerrorsdf["CoordY"] = coordsdf.iloc[:,1]
    
    percerrorsdf["NumGaps"] = numrows - desc.iloc[0,:]
    percerrorsdf["PercentageGaps"] = (1 - desc.iloc[0,:]/numrows)*100
    
    return percerrorsdf

#%%
def fillgapslr(nonfilleddata: pd.pandas.core.frame.DataFrame, coords: pd.pandas.core.frame.DataFrame, filtaux = pd.pandas.core.frame.DataFrame, work_w_filter = True, Use_distance = True, nummaxcorrec = 50, Use_t = True):
    
    '''Function to fill the gaps of time-series (originally rainfall) based on LR (multiple or single):
        
    Inputs
    ------------------
    nonfilleddata: dataframe[n x y]: 
        Dataframe with the time-series to be filled. Gaps should be stored as np.nan, 
        each column representing one point (e.g., rain-gauges, wells), with unique codes, and the index must be set as as datetime.
    
    coords: dataframe[y x 2]: 
        Dataframe with its index as the same codes as the df columns, plus a X and Y gepgraphic coordinates. 
    
    **filtaux: dataframe[y x 2]: 
        Dataframe with at least a column where the rows will represent each code that will be eventually be filled. 
        
    Parameters
    ------------------    
    work_w_filter: Set if you are working with a filter (filtaux) or not. Default is "True".
    Use_distance: You can define if the MLR will be ade using all available data as indendent variables of the MLR, or if only a 
        maximum number will be set. This makes sense if you are working with a large area, where even though high-correlations, will not have 
        physical meaning anymore, and therefore, should have the indepent variables set minimized. Default is "True".
    nummaxcorrec: It will be used if Use_distance is set to True, it is solely the maximum number of points used as independent variables. 
        Default is set to 50.
    Use_t: You can choose to use or not the t-statistic as also a limitant. Default is "True".
       
    Returns
    --------------------
    A pandas.DataFrame [n x z] with the filled time-series.
    An numpy.Array [m x z] with the columns as respectivelly mean, minimum and maximum for each point filled. 
    '''
    
    #Maximum number of gauges used for correction. This only makes sense if you want to use the distance between the wells
    #as a limitant for your MLR, i.e., if you want to use only the n-closest wells for you MLR.
    # At this point you can choose if you take into consideration the distance as a limitant factor, or not
    # In addition, you can choose to use or not the t-statistic as also a limitant
    
    # There were some issues while using the "True" or "False" statement for the t-statistics, therefore we are converting to a string "Yes" or "Not" inside the code, 
    # since it is more user-friendly the "True/False" statement for the users. 
    
    if Use_t == True:
        Use_t = "Yes"
    else:
        Use_t = "Not"
    
    if work_w_filter == True:
        work_w_filter = "Yes"
    else:
        work_w_filter = "Not"
    
    if Use_distance == True:
        Use_distance = "Yes"
    else:
        Use_distance = "Not"
    
    numcolumns = nonfilleddata.shape[1] # Number of points (columns)

    # For the case that you are working with a sub-set of your main dataset, you can define a filter with the respective points (columns)
    # that will be corrected. 
    #This reduces computation time when you are not interested on filling all the points available:
    
    if work_w_filter == "Yes":
        id_filter = filtaux.iloc[:,0] 
        num_corrigir = len(id_filter)
        nonfilleddata_corrigir = nonfilleddata[id_filter]
    else:
        num_corrigir = numcolumns
        nonfilleddata_corrigir = nonfilleddata

    #Create a distance matrix:
    #Convert the coords to a numpy matrix
    coords_np = coords.iloc[:, 0:2].to_numpy()
    dist_mat = cdist(coords_np, coords_np,metric = 'euclidean')

    #Convert again for a dataframe
    dist_mat_df=pd.DataFrame(dist_mat)
    dist_mat_df.columns=coords.index
    dist_mat_df.index=coords.index

    #Dataframe that will be filled
    filleddata = nonfilleddata_corrigir.copy()
    
    r2list = np.empty([num_corrigir, 3])
    r2list[:] = np.nan   
    #%% 
    # The loop will work for each point to be corrected

    for i in range(num_corrigir):
        name = nonfilleddata_corrigir.columns[i] #Point's name
        index = nonfilleddata_corrigir[name].index[nonfilleddata_corrigir[name].apply(np.isnan)] #Indexes of the point with gaps
        
        counteri =  0            
        r2listaux = np.empty([len(index),])
        r2listaux[:] = np.nan        
        
        
        #This loop will correct each day with gap in the point
        for j in tqdm.tqdm(index):

            
            #Pay attention that only points with no failures at the day to be corrected in the point to be corrected can be used for the model creation and regression
            names_0_that_day = nonfilleddata.columns[nonfilleddata.loc[j].apply(np.isfinite)] #Code of the points with no errors at this day
            num_ava_points = len(names_0_that_day)
            
            # If there are no points with zero failures at that day, the filling cannot take place
            if num_ava_points == 0:
                filleddata[name].loc[j] = np.nan
            
            else:
                if Use_distance == "Yes":
    
                    # Depending on the nummaxcorrec that you defined, it is possible that the number of available points to be used for corrrection is smaller than this maximum number, therefore 
                    # the code will have to use only the available points
                    if num_ava_points >= nummaxcorrec:
                        nclosest = dist_mat_df[names_0_that_day].loc[name].nsmallest(nummaxcorrec)
                    else:
                        nclosest = dist_mat_df[names_0_that_day].loc[name].nsmallest(num_ava_points)
            
                    # Name of the closest points to be used for correction
                    names_closest = nclosest.index
                else:
                    names_closest = names_0_that_day             
                    
                 
                # Matrix within the [X y] format
                datamatrix = nonfilleddata[names_closest].join(nonfilleddata[name])
                
                # Rows with NaN in either of our matrix cannot be used for regression, therefore we must drop them
                
                ###################################################################################################################
                # Extra part to solve the potential issue of the 1x2 or 0x0 matrices 
                # These lines are used to avoid the use of points with too many failures during the usable period of MLR built
                # which may at the end generate even 0x0 matrices if we do the drop imediatlly. 
                # Imagine that you have a point which has only failures in the period of observed data in the well to be corrected
                # except for the day when you want to fill. If you just use drop you will end up with a matrix 1x2 matrix or even 0x0 depending on the other points
                
                datamatrix = nonfilleddata[names_closest].join(nonfilleddata[name])
                datamatrix.dropna(subset=[name],inplace = True)
                # The drop is done first in the rows with NaNs in the failure day of the point to be corrected.  
                
                # Then the number of failures in the period of the first drop is computed:
                num_failures = pd.DataFrame(index = datamatrix.columns)
                num_failures["failures"] = 1 - datamatrix.count()/len(datamatrix)
                
                # A filter is applied:
                filter2 = num_failures <= 0.5
                datamatrix = datamatrix.loc[:, filter2.values]
                
                # And now the NaNs of the other points are droped. In this way we avoid to use a columns with more than 50% of
                # failures, which will damage our further MLR. 
                datamatrix.dropna(inplace = True)
                ##################################################################################################################
                
                # The len of the matrix is calculated
                len_mat = len(datamatrix)
                
                # If the len of the matrix is lower than 0 (that can happen), it is not possible to fill the gap
                if len_mat <= 1:
                    filleddata[name].loc[j] = np.nan
                
                # Else, the calculation can proceed
                else:
                    # Save the y and X
                    y = datamatrix.iloc[:,-1]
                    X = datamatrix.iloc[:,:-1]
                    # If the y column is formed only with 0s, it is not possible to proceed
                    if (y != 0).any(axis=0):
                        
                        # In addition, columns (names-X) that have only 0 as measurements are as well deleted
                        X = X.loc[:, (X != 0).any(axis=0)]
                        names_closest = X.columns #And the names of the ones used for correction are also updated
                        
                        # If the len of the matrix is lower than 0 (that can happen), it is not possible to fill the gap
                        if len(names_closest) < 1:
                            filleddata[name].loc[j] = np.nan
                        else:
                            #%%Multiple linear regression
                            regr = linear_model.LinearRegression()
                            regr.fit(X, y)
                            
                            #Maybe the matrix will remain as singular even without the 0 columns, thus, we have to test, and if this is the case, we can proceed the calculation do not taking into consideration the |t-stats| <=2
                            newX = pd.DataFrame({"Constant":np.ones(len(X))}, index = X.index).join(X)
                            if Use_t != "Yes" or np.linalg.det(np.dot(newX.T,newX)) == 0:
                                # Gaps filling
                                filleddata[name].loc[j] = regr.predict([nonfilleddata[names_closest].loc[j]])
                                
                                # R2 computation:
                                r2listaux[counteri] =  regr.score(X, y)
                                
                            else:
                                #t-statistics
                                params = np.append(regr.intercept_,regr.coef_)
                                predictions = regr.predict(X)
                    
                    
                                newX = pd.DataFrame({"Constant":np.ones(len(X))}, index = X.index).join(X)
                                
                                # There is the really small, but still existent risk of the num of cols being the same number of
                                # rows, in this case, we need to set the diference manually to 1 and avoid an error:
                                # Remember though that if this ever happen, it means that your model is most likely overfiting,
                                # because it has as much variables as the number of observations!
                                
                                if len(newX)-len(newX.columns) != 0:
                                    MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))
                                    
                                else:
                                    MSE = (sum((y-predictions)**2))/ 1 
                                 
                                    
                                var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
                                sd_b = np.sqrt(var_b)
                                ts_b = params/ sd_b
    
                    
                                myDF3 = pd.DataFrame()
                                myDF3["Coefficients"],myDF3["t values"]= [params,ts_b]
                    
                    
                                myDF4 = myDF3[1:] 
                                myDF4 = myDF4.set_index(names_closest)
        
                                filt = ((myDF4['t values'].abs() >= 2))
                                new_names = myDF4.loc[filt].index
                    
                                # We tested if there are points with abs(t-stats) < 2, and if yes, we re-calculate MLR:
                                if len(new_names) == len(names_closest):
                                    # Gaps filling
                                    filleddata[name].loc[j] = regr.predict([nonfilleddata[names_closest].loc[j]])
                                            
                                    # R2-computation:
                                    r2listaux[counteri] =  regr.score(X, y)                   
                                
                                else:
                                    # Matrix within the [X y] format
                                    datamatrix_new = nonfilleddata[new_names].join(nonfilleddata[name])
                                    # Rows with NaN in either of our matrix cannot be used for regression, therefore we must drop them
                                    datamatrix_new.dropna(inplace = True)
                     
                                    # Save the y and X
                                    y_new = datamatrix_new.iloc[:,-1]
                                    X_new = datamatrix_new.iloc[:,:-1]
                                    # If the y column is formed only with 0s, it is not possible to proceed
                                    if (y_new != 0).any(axis=0):
                     
                                        # In addition, columns (names-X) that have only 0 as measurements are as well deleted
                                        X_new = X_new.loc[:, (X_new != 0).any(axis=0)]
                                        new_names = X_new.columns #And the names of the ones used for correction are also updated
                                        # If the len of the matrix is lower than 0 (that can happen), it is not possible to fill the gap
                                        if len(new_names) < 1:
                                            filleddata[name].loc[j] = np.nan
                                        else:
                                            #%%Multiple linear regression
                                            regr_new = linear_model.LinearRegression()
                                            regr_new.fit(X_new, y_new)
                                            
                                            # R2-computation:
                                            r2listaux[counteri] =  regr_new.score(X_new, y_new)
                                            
                                            # Gaps filling
                                            filleddata[name].loc[j] = regr_new.predict([nonfilleddata[new_names].loc[j]])   
                                    else:
                                        filleddata[name].loc[j] = np.nan 
                    else:
                        filleddata[name].loc[j] = np.nan        
             
            counteri = counteri + 1
            
            # For each column, we gather the mean, min and max R2-computed. 
            r2list[i, 0] = np.nanmean(r2listaux)
            r2list[i, 1] = np.nanmin(r2listaux)
            r2list[i, 2] = np.nanmax(r2listaux)
            
    # It is possible that negative values will be calculated, thus we replace them per 0:
    filleddata[filleddata < 0.1] = 0
    
    r2listdf = pd.DataFrame(index = filleddata.columns, columns = ["Mean","Min","Max"], data = r2list) 
    
    return filleddata, r2listdf

#%%
def plotgaps(summarygapsstations: pd.pandas.core.frame.DataFrame, crsproj = 'epsg:4326', backmapproj = True, figsizeproj = (15, 30), cmapproj = "Reds"):
    """
    Inputs
    ------------------
    summarygapsstations: dataset[y x 4]: 
        The same dataframe output from the fillinggaps.summarygaps function.
    
    Returns
    --------------------
    plt.plot: The output is a plt.plot with the points spatially distributed in the area, and with a legend bar 
        showing the percentage of gaps (from 1 to 100). A background map can be also shown if your coordinate system 
        "crsproj" is set to 'epsg:4326'.
        
    """
    if backmapproj == True:
        
        if crsproj == 'epsg:4326':
        
            crs = {'init': crsproj}
            geometry = [Point(xy) for xy in zip(summarygapsstations["CoordX"], summarygapsstations["CoordY"])]
            geodata=gpd.GeoDataFrame(summarygapsstations,crs=crs, geometry=geometry)
            geodatacond = geodata

            # The conversiojn is needed due to the projection of the basemap:
            geodatacond = geodatacond.to_crs(epsg=3857)

            # Plot the figure and set size:
            fig, ax = plt.subplots(figsize = figsizeproj)

            #Organizing the legend:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)

            #Ploting:
            geodatacond.plot(ax=ax, column='PercentageGaps', legend=True, cax = cax, cmap = "Reds")
            cx.add_basemap(ax)
        
        else:
            crs = {'init': crsproj}
            geometry = [Point(xy) for xy in zip(summarygapsstations["CoordX"], summarygapsstations["CoordY"])]
            geodata=gpd.GeoDataFrame(summarygapsstations,crs=crs, geometry=geometry)
            geodatacond = geodata

            # Plot the figure and set size:
            fig, ax = plt.subplots(figsize = figsizeproj)

            #Organizing the legend:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
        
            #Ploting:
            geodatacond.plot(ax=ax, column='PercentageGaps', legend=True, cax = cax, cmap = cmapproj)
        
    else:
        crs = {'init': crsproj}
        geometry = [Point(xy) for xy in zip(summarygapsstations["CoordX"], summarygapsstations["CoordY"])]
        geodata=gpd.GeoDataFrame(summarygapsstations,crs=crs, geometry=geometry)
        geodatacond = geodata

        # Plot the figure and set size:
        fig, ax = plt.subplots(figsize = figsizeproj)

        #Organizing the legend:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        
        #Ploting:
        geodatacond.plot(ax=ax, column='PercentageGaps', legend=True, cax = cax, cmap = cmapproj)   
            
    
    return plt.show()










