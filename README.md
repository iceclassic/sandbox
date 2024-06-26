# Nenana Ice Classic 2024

# Table of Contents

1. [Data](#data)
    <!--  -->
    1.1. [Loading](#loading-and-inspecting-the-data)

    1.2. [Plotting](#plotting-tips-and-tricks)
2. [Notebooks](#notebooks)

    2.1. [Data Viewer](#data_vieweripynb)

    2.2. [PreProcessing](#data_vieweripynb)

    2.3. [Prediction of Ice thickness](#Prediction_of_thicknessipynb)

    2.4  [Ice Cam](#Ice_cam.ipynb)
3. [Results](#results)
4. [Discussion](#discussion)
5. [Conclusion](#conclusion)

## Data

We have compiled relevant data in the file `Time-Series_DATA.txt`
The file is composed of two parts:

1. Sources and metadata

- Description and source for every column
- Explanation of modification of original data when necessary

2. Data

- Comma-separated values of the data
- At the moment the contents of the files are:
  - Air temperature
  - Predicted ice thickness
  - Rainfall
  - Snowfall
  - Snow depth
  - Water temperature
  - Discharge
  - Air temperature
  - Wind speed
  - Ice thickness
  - Cloud coverage
  - Solar radiation
  - (Nearby) Glacial temperature, mass balance, precipitation
  - Global climate oscillation parameters

### Loading and inspecting the Data

As we are working with time-series it is recommended to load the data as a dataframe with the following command

```python
Data=pd.read_csv('Time_series_DATA.txt',skiprows=149,index_col=0)`
Data.index=pd.to_datetime(Data.index):
```

The ```skiprows```, is included to avoid loading the sources/licences and metadata of our individual timeseries.
A quick inspection of the data can now be easily made with

```python
Data.info()
```

The module `make_plots.py`contains some useful function that may help to explore/inspect the contents of the data.

In particular

- `explore_data(Data)`

    Makes a plot for each column in the dataframe.

- `interactive_data(Data)`

    Makes an interactive plot with some of the most relevant columns

Alternatively, we can use the function `numpyfy` to convert the dataframe to a numpy array.

### Plotting tips and tricks

Using pandas it is really easy to plot specific columns by simply specifying the column name.

```python
plt.plot(Data.index,Data['colname'])
```

The index of Data correspond to the datetime-object `YYYY-MM-DD` associated with that value.

Alternatively it can be useful to use another x-axis, we have multiple options

1. `Data.index.year`
2. `Data.index.month`
3. `Data.index.day`

 But more useful for our use-case we may need normalize dates such as

 1. `Data['Days since start of year']`
 2. `Data['Days to break up ']`

 In some particularly sparse columns such as `Data['IceThickness [cm]']`
 the simple `plt.plot`command does not work properly ( plt.scatter works just fine).

  In this cases the solution is to create a copy,  eliminate `nan` values so the df is no longer sparse.

 ```python
 Data_ice= Data.dropna(subset=['IceThickness [cm]']).copy()
 plt.plot(Data_ice.index,Data_ice['IceThickness [cm]'])
 ```

## Notebooks

### `Data_viewer.ipynb`

 Notebook with a multitude of p
 lots that characterize the contents of the dataframe. USeful to get a feeling for the contents of `Time_series_DATA.txt`

### `PreProcessing.ipynb`
Notebook with plot that more relevant to try to uncover tendencies or relationship between the variables that may allow to predict the data

### `Prediction_of_thickness.ipynb`
Notebook were the Ashton model is implemented to predict ice thickness in term of temperatures

### `Ice_cam.ipynb`
Notebook use to create a video and very light processing of the picture obtained form [Nenana Ice Classic webpage](https://www.nenanaakiceclassic.com/)

The resulting videos can be found [here](https://www.youtube.com/playlist?list=PLo0kgRXad08IIh6dewL-pxnd7zaNE-dhL)