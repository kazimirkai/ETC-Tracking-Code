#!/usr/bin/env python
# coding: utf-8
import glob
import xarray as xr
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
import itertools
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.collections import LineCollection
from scipy.spatial import distance
import time
from datetime import datetime, timedelta
from pyproj import Proj
from pathlib import Path 

# ----------------------------
# PARAMETERS
# ----------------------------
data_dir = '/network/rit/lab/basulab/Projects/DFS/DATA/EDDEv2/EDDEv2_PRESSURE/6Hourly/SLP_SSP3-7.0/2025-2100'
pressure_var = 'SLP'

# Tracking sensitivity
dist_threshold_M = 700_000        # maximum distance between points (m)
dist_threshold_m = 12_000         # minimum distance between points (m)
neighborhood_m = 200_000          # neighborhood for mean SLP (m)
min_pressure_diff = 1.5           # hPa above neighborhood mean
max_turn_angle = 60               # degrees
min_track_length = 4


# Time selection
Year = 2050                             #define year of interest
start_date = datetime(Year, 1, 1)       #define start date based on year
end_date   = datetime(Year, 12, 31)     #define end date based on year 
output_csv = f"{Year}.csv"              #defines output based on year 

# Plotting domain       
x_min = -98
x_max = -45                                #bound the domain in the eastern US 
y_min = 25
y_max = 55

bounds_pressure = np.arange(950, 1010, 5)   #set initial detection pressure bounds
max_slp= 1013.0                             #set maximum pressure bound post-genesis


# ----------------------------
# Define subdirectories for outputs
# ----------------------------
base_dir = Path.cwd()                                  # base directory 

csv_dir = base_dir / "multi_output_csvs"
id_png_dir = base_dir / "multi_ID_plots"              #defining directories to input outfiles
slp_png_dir = base_dir / "multi_slp_plots"
density_png_dir = base_dir / "multi_density_plots"

csv_dir.mkdir(exist_ok=True)
id_png_dir.mkdir(exist_ok=True)                             # Create directories if they don't exist
slp_png_dir.mkdir(exist_ok=True)    
density_png_dir.mkdir(exist_ok=True)

output_csv = csv_dir / f"{Year}_multi.csv"
output_id_png = id_png_dir / f"{Year}_tracks_by_ID.png"          # Output file paths (now in subdirectories)
output_slp_png = slp_png_dir / f"{Year}_tracks_by_slp.png"
output_density_png = density_png_dir / f"{Year}_track_density.png"

# ---------------------------------------------
# Function to calculate angles between vectors 
# ---------------------------------------------

def angle_between_vectors(p1, p2, p3):           #define a function that takes the input of three vector points (current and prev two)

    v1 = np.array([p2['x'] - p1['x'], p2['y'] - p1['y']])      #creates a vector from the first to second point 
    v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])      #creates a vector from the second to current point 

    # Normalize
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:   #return no angle if the vector directions are the same (linalg.norm returns just angle)
        return 0  

    v1_u = v1 / np.linalg.norm(v1)              #returns unit vectors 
    v2_u = v2 / np.linalg.norm(v2)

    # Compute angle in degrees
    dot = np.clip(np.dot(v1_u, v2_u), -1, 1)    # filters out bad angle results from rounding errors 
    angle_rad = np.arccos(dot)                  #returns angles between unit vectors
    angle_deg = np.degrees(angle_rad)           #conv to degrees

    return angle_deg                            #returns track angle change
    
# -------------------------------------------------------------------
# Function to accumulate, link, and color tracking points identified
# -------------------------------------------------------------------
def line_segments(df, x='x', y='y', slp='slp', bounds=None, transform = ccrs.LambertConformal(    #define line segment formation function to take lat/lon and pressure 
    central_longitude=-97.0,
    central_latitude=36.775,                                          #input parameters for map projection 
    standard_parallels=(33, 45) 
)):
    points = df[[x, y]].to_numpy().reshape(-1, 1, 2)                 
#converts the lat/lon portion of the points dataframe to a numpy array for faster computation (-1 implies an index, the middle dimension is for connecting rows, and the '2' holds the x/y)               
    if len(points) < 4:                                               #if the number of points in the track is less than 4
        return None                                                   #function does not output segments
    segments = np.concatenate([points[:-1], points[1:]], axis=1)      #define the segment variable concotenate points into 3d matrices of the same shape as the points metrix                          
    cmap = plt.get_cmap('rainbow')                                    #define colormap 
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N) if bounds is not None else mpl.colors.Normalize(    #normalize pressure values into discreet color bins according to set bounds
        vmin=df[slp].min(), vmax=df[slp].max()
    )
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2.5, transform=transform)   #define linecollection dataframe (matplotlib) from all line properties
    lc.set_array(df[slp].to_numpy())                                  #convert linecollection dataframe to numpy 
    return lc                                                         #return it from function

# ----------------------------------------------------------
# Load netCDF monthly files according to daterange provided 
# ----------------------------------------------------------
all_files = set()
current = start_date
while current <= end_date:
    year = current.strftime('%Y')
    matched_files = glob.glob(f"{data_dir}/{year}/*{current.strftime('%Y-%m')}*.nc")
    all_files.update(matched_files)
    current += timedelta(days=1)

all_files = sorted(all_files)
if not all_files:
    raise FileNotFoundError("No NetCDF files found for the specified date range.")

ds = xr.open_mfdataset(all_files, combine='by_coords', engine='netcdf4')

# ---------------------------------------
# Restrict domain to desired x/y bounds
# ---------------------------------------
proj = Proj(                                    # Define the projection and it's parameters
    proj='lcc',
    lat_1=33,
    lat_2=45,
    lat_0=36.775,
    lon_0=-97.0,
    R=6370000)

x_min_proj, y_min_proj = proj(x_min, y_min)    # Convert bounding box from lat/lon to x/y given the preset bounds
x_max_proj, y_max_proj = proj(x_max, y_max)

xs = ds["x"]                                       # Define original dataset coords
ys = ds["y"]

xmask = (xs >= x_min_proj) & (xs <= x_max_proj)    # Create a mask using x/y coordinates from bounds
ymask = (ys >= y_min_proj) & (ys <= y_max_proj)

x_idx = np.where(xmask)[0]                   #find the index of the masked coord bounds 
y_idx = np.where(ymask)[0]

ds = ds.isel(x=x_idx, y=y_idx)       # Redefine original dataset just including those indexes 

xs = ds["x"]                     # Update x/y labels for the dataset 
ys = ds["y"]

# ----------------------------
# Select pressure variable
# ----------------------------
slp = ds[pressure_var]/100          #define slp variable from netcdf file and convert to hPa
slp = slp.sel(time=slp['time.month'].isin([11, 12, 1, 2, 3]))   #select for winter months 

# ----------------------------
# SUBSAMPLE 6-HOURLY (not necessary now that i'm using 6-hourly data by default)
# ----------------------------
#subsample_hours=6
#times = slp.time.values
#slp = slp.sel(time=times[::subsample_hours//int((times[1]-times[0])/np.timedelta64(1, 'h'))])

#print(f"Using time range: {slp.time.values[0]} → {slp.time.values[-1]}")
#print(f"Total timesteps: {slp.time.size}")

# ------------------------------
# SIMULTANEOUS TRACKING WITH PROGRESS REPORT
# ------------------------------
tracks = []          # list of tracks
active_tracks = []   # points for the active track  

start_time = time.time()                
              #define empty track, total steps (calculated in previous section), and start time of tracking program
total_steps = slp.time.size              

print("SLP min/max:", slp.min().values, slp.max().values)  #print min max in first step 
print("Time steps:", slp.time.size)                        #print total time steps 

for t in range(total_steps):                                            #for each time step, t 
    step_start = time.time()                                            #define the start of the step 
    slp_snapshot = slp.isel(time=t)                                     #take a field of slp vaues at this time t
    smoothed = gaussian_filter(slp_snapshot.values, sigma=2)            #apply gaussian filter and define field 

    coords = peak_local_max(-smoothed, min_distance=3)       #find coordinates of all local minima. Anything within 150km of a low
                                                              #is considered part of the same minima
    pmins = []
    for c in coords:                                                 #for each local pressure minimum 

        pval = float(slp_snapshot.isel(y=c[0], x=c[1]).values)       #get slp of min
        SN = float(ys[c[0]])                                         #get lat of min
        WE = float(xs[c[1]])                                         #get lon of min

        dy = np.abs(ys - SN) * 111_000                                # 1° latitude ≈ 111 km
        dx = np.abs(xs - WE) * 111_000 * np.cos(np.radians(SN))       # correct longitude scaling
            #get distances from min to all other grid pts
        neighborhood = slp_snapshot.where((dy <= neighborhood_m) & (dx <= neighborhood_m), drop=True)      
                                            #If distance less than neighborhood distance, acquire all those coords 
        
        #print(f"Candidate min at x={WE}, y={SN}, SLP={pval}")    #debugging prints
        #print("Neighborhood mean:", neighborhood.mean().values)
        
        if neighborhood.mean().values - pval > min_pressure_diff:
            pmins.append({'y': SN, 'x': WE, 'slp': pval})
 #take average of slp within neighborhood and compare to minimum to filter out insignificant minima, append significant minima to pmins array
                                          
    used_minima = set()

    # --------------------------------
    # Link existing tracks
    # --------------------------------
    for track in active_tracks:                 #for any given track in the current active tracks
        prev = track['points'][-1]              #defining the previous point in the track
        candidates = []                         #defining an empty array for new point candidates to go in

        for i, p in enumerate(pmins):           #for all detected minima 
            if i in used_minima:                #if is already used in another track
                continue                        #skip for the application of the next section

            d = np.sqrt((p['x'] - prev['x'])**2 + (p['y'] - prev['y'])**2)   #calculate distance from previous point to this one
            if d <= dist_threshold_M and d >= dist_threshold_m:            #if it's within the predefined distance thresholds... 
                candidates.append((i, p, d))        #add the id, pressure dictionary point (x/y/slp), and distance


        if candidates:                               #search among candidates that fit the previously defined criteria
            valid_candidates = []                    #set empy candidates array 

            for i, p, d in candidates:               #acquiring index, coords, pressure, and distance of new point    
                
                # ---- ABSOLUTE PRESSURE GATE ----
                if p['slp'] > max_slp:               #max pressure check
                    continue                         #skip to next canditate 

                # ---- TURN ANGLE FILTER ----
                if len(track['points']) >= 2:        #if length of track is at least two points
                    angle = angle_between_vectors(track['points'][-2], prev, p)   #call max angle function 
                    if angle > max_turn_angle:       #if angle is too large 
                        continue                     #skip to skip to next candidate 

                valid_candidates.append((i, p, d))   #all other points (that havent skipped), append 
 
            if not valid_candidates:                 #for not valid points (termination block)
                track['last_used'] = False           #label last used point as invalid
                continue                             #move to track finalization 

            # choose best after all filters to reduce candidates slipping through threshold condition logic 
            i_best, p_best, _ = min(valid_candidates, key=lambda x: x[1]['slp'])   #for valid points, find lowest pressure candidate

            track['points'].append(p_best)           #add point
            track['last_used'] = True                #state that the track is continuing 
            used_minima.add(i_best)                  #confirm that the last point in the track has been used (can't start a new track)

        else:                                        #if no minima are found at all
            track['last_used'] = False               #end track     
            print(f"No valid minima found at timestep {t}")    #state a track is ending 

    # --------------------------------
    # Finalize tracks that ended
    # --------------------------------
    still_active = []                                      #make array for still active tracks

    for track in active_tracks:                            #for each active track...
        if track['last_used']:                             #if a point was used in this timesetp
            still_active.append(track)                     #add track to the 'still active' array
        else:                                              #if a point was not used... 
            if len(track['points']) >= min_track_length:   #and if the track was long enough...
                tracks.append(track['points'])             #add track to 'tracks' array

    active_tracks = still_active                           #redefine the active tracks with only the surviving ones 

    # --------------------------------
    # Start new tracks from unused minima
    # --------------------------------
    for i, p in enumerate(pmins):                #for other track candidates 
        if i not in used_minima:                 #if theyre as of yet unused (new)
            if p['slp'] > max_slp:               #max pressure check 
                continue                         #if pressure too high, dont append point
            active_tracks.append({               #add to active tracks 
                'points': [p],                   #create points array in new tracks 
                'last_used': True})              #confirm the points added are being used 
        
   # Progress print
    step_end = time.time()                                                        #acquire utc time what each time step finished
    elapsed = step_end - start_time                                               #calculate how much time since program started 
    avg_per_step = elapsed / (t + 1)                                              #average the time on each step, including this one
    remaining = avg_per_step * (total_steps - t - 1)                              #extrapolate the time left in the program
    if t % 10 == 0 or t == total_steps - 1:                                       #every 10 steps or if it's the last step...
        print(f"[{t+1}/{total_steps}] step time: {step_end-step_start:.2f}s, "     #print which step the program is on, how long it took
              f"elapsed: {elapsed:.2f}s, estimated remaining: {remaining:.2f}s")   #how much time has gone by since the program started, how much remains

# --------------------------------
# Catch tracks still active at end
# --------------------------------
for track in active_tracks:                         #for any track still active at the end of the time frame
    if len(track['points']) >= min_track_length:    #if the length of the track is long enough...
        tracks.append(track['points'])              #add to to the tracklist

# ---------
# SAVE 
# ---------
rows = []                                           #create array for the output csvs 
for tid, track in enumerate(tracks):                #get each track (tid) and list of points (track) in the tracklist (tracks)
    for p in track:                                 #for each trackpoint 
        rows.append({**p, 'track_id': tid})         #add its data (comes with x/y and pressure data), append the id each point belongs to as another following column 

df = pd.DataFrame(rows)                             #create/define a data frame of the "rows" array 
df.to_csv(output_csv, index=False)                  #convert the dataframe to a csv with the title specified earlier by 'output_csv' variable
print(f"Saved CSV: {output_csv}")                   #say so 

#--------------------
# Regular Track Plot 
#--------------------

# Load the CSV
df = pd.read_csv(output_csv)                        #define and read tracking output file 

# Convert x/y Lambert Conformal to lat/lon
proj_lcc = Proj(proj='lcc', lat_1=30, lat_2=60, lat_0=36.775, lon_0=-97)         #specify and define paremeters of projection to use 
df['lon'], df['lat'] = proj_lcc(df['x'].values, df['y'].values, inverse=True)    #label the x and y values from the outfile as 'lat' and 'lon' for later reference

# Colormap for track IDs
track_ids = df['track_id'].unique()                                       #acquire/define the list of unique track ids
cmap = plt.get_cmap('tab20', len(track_ids))          #define colormap to use, adjust the number of colors to number of unique tracks (far most than 20 in most cases here)
norm = mpl.colors.BoundaryNorm(np.arange(len(track_ids)+1)-0.5, cmap.N)   #Normalize the colormap values to be in beteen integer +/- 0.5 so the ids clearly lie within each color 

fig = plt.figure(figsize=(10,8))                                #define figure, size
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())    #number the first plot being made in this script

# Map extent and features
ax.set_extent([-100, -50, 20, 65], crs=ccrs.PlateCarree())      
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)                                 #define extent, projection, and aesthetic features/colors of map 
ax.add_feature(cfeature.STATES, linewidth=0.4)
ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)

# Plot each track with a different color
for tid in track_ids:                                                     #for each track (id) 
    track_df = df[df['track_id'] == tid]                                  #take out the rows that belong to that track (id)
    points = track_df[['lon','lat']].to_numpy().reshape(-1,1,2)           #define the points in the track as an array of the latitude and longitudes 
    if len(points) < 2:                                                   #if the length of the track is less than 2 (shouldn't be the case due to prior filtering) 
        continue                                                          #move to next track
    segments = np.concatenate([points[:-1], points[1:]], axis=1)          #defines the segments as pairs of consecutive points by combinine matching indeces from two separate tuples (all by last point, all but first point) 
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2.0)   #creates line collection usiung a matplotlib tool from all the segments using the earlier defined colormap and normalizations
    lc.set_array(np.full(len(track_df)-1, tid))                           #assigns color values to all segments and 
    ax.add_collection(lc)                                                 #adds the line collection to the plot 

# Scatter all points (optional)
ax.scatter(df['lon'], df['lat'], c=df['track_id'], cmap=cmap, s=25, alpha=0.8, transform=ccrs.PlateCarree(), norm=norm)
                               #creates a scatter plot of all points according to their lat/lons with color scalar value, c, marker size 25, transparancy 0.8, and projection platecaree
# Colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)                        #maps track ids to colors using earlier specified colormaps and normalizations usng mpl scalarmappable tool
sm.set_array([])                                                        #create an empty list for actual data to go into 
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', ticks=track_ids) #creates colorbar using scalarmappable tool, with various specifications, ticks for each track id
cbar.set_label('Track ID')                                              #labels it 

plt.title("Tracked Low-Pressure Centers — Colored by Track ID")         #title the plot 
plt.savefig(output_id_png, dpi=300, bbox_inches='tight')                #save it with 300 dpi size, small borders
plt.close()                                                             #close plot creation

print(f"Saved ID plot: {output_id_png}")                                #anounce that this plot was created
# ------------
# SLP PLOT
# ------------

# Colormap for pressures (lower pressure = darker)
cmap_slp = plt.get_cmap('viridis_r')  

# Compute segment mean pressures for normalization
segment_slps = []
for tid, track_df in df.groupby('track_id'):
    slp_vals = track_df['slp'].to_numpy()
    if len(slp_vals) < 2:
        continue
    # mean pressure for each segment (between consecutive points)
    segment_slps.extend(0.5 * (slp_vals[:-1] + slp_vals[1:]))

segment_slps = np.array(segment_slps)
norm_slp = mpl.colors.Normalize(vmin=segment_slps.min(), vmax=segment_slps.max())

# Create figure
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Map features
ax.set_extent([-100, -50, 20, 65], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES, linewidth=0.4)
ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)

# Plot each track
for tid, track_df in df.groupby('track_id'):
    if len(track_df) < 2:
        continue

    points = track_df[['lon','lat']].to_numpy().reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    slp_vals = track_df['slp'].to_numpy()
    segment_slp = 0.5 * (slp_vals[:-1] + slp_vals[1:])

    lc = LineCollection(
        segments,
        cmap=cmap_slp,
        norm=norm_slp,
        linewidths=2.0,
        transform=ccrs.PlateCarree()
    )
    lc.set_array(segment_slp)
    ax.add_collection(lc)

# Scatter points (optional, all black for clarity)
ax.scatter(
    df['lon'],
    df['lat'],
    c='k',
    s=15,
    alpha=0.6,
    transform=ccrs.PlateCarree(),
    zorder=3
)

# Colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap_slp, norm=norm_slp)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
cbar.set_label('Segment Mean SLP (hPa)')

plt.title("Tracked Low-Pressure Centers — Colored by Mean Segment SLP")
plt.savefig(output_slp_png, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved slp plot: {output_slp_png}")

# -----------------------------
# Track Density Contour Plot
# -----------------------------
# User parameters
smoothing_sigma = 6.0
grid_resolution = 200
density_cmap = "Blues"
buffer_km_density = 300
num_contours = 16

x_all = df['lon'].values
y_all = df['lat'].values

x_min_d, x_max_d = x_all.min(), x_all.max()
y_min_d, y_max_d = y_all.min(), y_all.max()

lat_buffer = buffer_km_density / 111.0
mean_lat = y_all.mean()
lon_buffer = buffer_km_density / (111.0 * np.cos(np.deg2rad(mean_lat)))

x_min_d -= lon_buffer
x_max_d += lon_buffer
y_min_d -= lat_buffer
y_max_d += lat_buffer

x_grid = np.linspace(x_min_d, x_max_d, grid_resolution)
y_grid = np.linspace(y_min_d, y_max_d, grid_resolution)

H, _, _ = np.histogram2d(y_all, x_all, bins=grid_resolution, range=[[y_min_d, y_max_d],[x_min_d, x_max_d]])
H_smooth = gaussian_filter(H, sigma=smoothing_sigma)

X, Y = np.meshgrid(x_grid, y_grid)
min_density = H_smooth[H_smooth>0].min()
max_density = H_smooth.max()
levels = np.linspace(0, 0.15, num_contours)

fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-100, -50, 20, 65], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES, linewidth=0.4)
ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)

cf = ax.contourf(X, Y, H_smooth, levels=levels, cmap=density_cmap, norm=mpl.colors.PowerNorm(gamma=0.5), extend='max')
ax.contour(X, Y, H_smooth, levels=levels, colors='k', linewidths=0.5)

cbar = plt.colorbar(cf, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label("Track Density (smoothed)")

plt.title("Track Density")
plt.savefig(output_density_png, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved density plot: {output_density_png}")
