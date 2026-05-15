These documents were created as part of an internship in contribution to a project entitled “Assessing Climate Exposure and Financial Impacts at the Census Tract Level in New York State" alongside fellow undergraduate, graduate, and faculty at UAlbany. The initiative is funded by the New York State Department of Financial Services. 

The .py is intended to track extratropical cyclones, in this case, using EDDEv2 dynamically downscaled climate data, accessed via AWS and run within the UAlbany Research IT system. Inputs are in the form of netcdf and output csvs contain latitude, longitude, pressure (hPa), and tracking id for each point (so consecutive points can be classified as the same track). The latter portion also outputs track maps colored by track id and by pressure at each point as well as a density plot. 

The minimum detection was based on another WRF reanalysis ETC tracking algorithm written by Philip Yeh, 
based on the method found in Crawford et al. 2021. The tracking (particularly multi-track functionality), graphic, and data analysis portions are custom 

The jupyter notebook documents the developmental process of the tracking algorithm as well as combined total, lysis, genesis density for both climate scenarios available in EDDEv2 (SSP2-4.5, SSP3-7) and for stronger (sub 980 hPa) storms. 

The input parameters for the algorithm limits tracks to a minimum of 4 points, confines the initial detection maximum to 1010 mb and the tracking (post-genesis) maximum to 1013 mb, the maximum turn angle between points to 60 degrees, minimum and maximum point-to-point distance thresholds to 12km and 700km respectively, neighborhood distance (search radius) to 200km, and pressure-minimum difference from the mean in the search radius to 1.5mb. These were defined by intuition, computational optimization, and prior literature. 

Unfortunately, due to time limits of the internship, these parameters have not been tuned with reanalysis data. Furthermore, there is some spurious clustering of track points over some apparently uncorrelated locations which should be resolved before any professional use of this program.  

With thay said, density plots demonstrate sensible behavior for ETCs, maximizing overall densities in the Canadian Maritimes and suggesting western weighted genesis densities and eastern weighted lysis densities. One notable finding is a very minor increase in the number of total tracks in the higher emission scenario (by 2.9%) but a decrease in the stronger, <980 hPa, storms (by 8.1%). In both scenarios, <980 hPa storms made up 8 to 9% of the total tracked systems.  
