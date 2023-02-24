# Geospatial analysis

This repository provides a series of functions and geospatial analysis focusing mainly on satellite grid and rain gauges precipitation data. 

The analysis performed with Satellite-grid data is organized as "1X_", and the part with the rain-gauges as "2X_". 

a) So far, regarding the grid data, (1) first it is made the conversion of data from a raster format to dataframes and a SPI computation per grid; and then (2) it is made a graphical and map visualization of the SPI dataset. 

b) Regarding the rain-gauges data, (1) first the rain-gauges precipitation time-series of several gauges are organized, and filtered considering a maximum gap threshould, and after further it is applied some filters to minimize potential measurement errors; after that (2) a gap filling module is applied to fill the time-series' missing values.  

