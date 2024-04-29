
import os
import sys
from timeit import default_timer as timer 
import datetime
now = datetime.datetime.now()

# Data manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.stats import norm,kstest

def explore_data(Data):
    for i, col in enumerate(Data.columns):
        col_data = Data[col].copy()
        col_data.dropna(inplace=True)
        if not col_data.empty: 
            plt.figure(figsize=(20, 3))
            plt.plot(col_data.index, col_data.values, label=col, color=plt.cm.tab10(i % 10))
            plt.legend()
            #plt.xlabel('Date')
            #plt.ylabel('Value')
            plt.title(col)
            plt.tight_layout()
            plt.show()

def interactive_data(Data):

    Data_clouds= Data.dropna(subset=['Regional: Cloud coverage [%]']).copy()
    Data_ice= Data.dropna(subset=['IceThickness [cm]']).copy()
    Data_solar= Data.dropna(subset=['Regional: Solar Surface Irradiance [W/m2]']).copy()

    # Initialize figure
    fig = go.Figure()

    # Add Traces ( plots elements)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='break up date', line=dict(color='red', width=0.4, dash='dot')))
    fig.add_trace(go.Scatter(x=Data.index,y=Data["Regional: Air temperature [C]"],name="Air temp",yaxis="y",line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=Data.index,y=Data["Predicted ice thickness [m]"]*100,name="Predicted ice thickness",yaxis="y4",line=dict(color='lime')))
    fig.add_trace(go.Scatter(x=Data_ice.index,y=Data_ice["IceThickness [cm]"],name="Ice thickness",yaxis="y4",line=dict(color='navy')))
    fig.add_trace(go.Scatter(x=Data.index,y=Data["Nenana: Rainfall [mm]"],name="Rainfall",yaxis="y2",line=dict(color='lightseagreen')))
    fig.add_trace(go.Scatter(x=Data.index,y=Data["Nenana: Snowfall [mm]"],name="Snowfall",yaxis="y2",line=dict(color='slateblue')))
    fig.add_trace(go.Scatter(x=Data.index,y=Data["Nenana: Snow depth [mm]"],name='Snow depth',yaxis="y2",line=dict(color='magenta')))
    fig.add_trace(go.Scatter(x=Data.index,y=Data['Nenana: Mean water temperature [C]'],name="Water temp",yaxis="y",line=dict(color='firebrick')))
    fig.add_trace(go.Scatter(x=Data.index,y=Data['Nenana: Mean Discharge [m3/s]'],name="Discharge",yaxis="y3",line=dict(color='dodgerblue')))
    fig.add_trace(go.Scatter(x=Data_clouds.index,y=Data_clouds["Regional: Cloud coverage [%]"],name="Cloud coverage",yaxis="y5",line=dict(color='slategray')))
    fig.add_trace(go.Scatter(x=Data_solar.index,y=Data_solar["Regional: Solar Surface Irradiance [W/m2]"],name="Solar Surface Irradiance",yaxis="y6",line=dict(color='orangered')))
    # dropdown menu to select which timseries to plot
    # fig.update_layout(
    #     updatemenus=[
    #         dict(active=0,
    #             buttons=list([
    #                 dict(label="All",method="update",args=[{"visible": [True, True, True,True,True, True, True,True,True,True]}]),
    #                 dict(label="Temperature",method="update",args=[{"visible":  [True, False, False,False,False, False,True,False,False,False]}]),
    #                 dict(label="Ice+Snow",method="update",args=[{"visible":    [False,True,True,False,True,True,False,False,False,False]}]),
    #                 dict(label="Discharge+Rain",method="update",args=[{"visible":   [False, False, False,True,False, False,False,True,False,False]}]),
    #                 dict(label="Clouds+Solar Radiation",method="update",args=[{"visible":   [False, False, False,False,False, False,False,False,True,True]}])]),)])


    break_up_times=pd.read_csv('https://github.com/GabrielFollet/ICE_data_dump/blob/main/BreakUpTimes.csv?raw=true')
    break_up_times.head()
    break_up_times['timestamp'] = pd.to_datetime(break_up_times[['Year', 'Month', 'Day']])  # want index wiht only date not time
    break_up_times['timestamps'] = pd.to_datetime(break_up_times['timestamp'])
    break_up_times.set_index('timestamp', inplace=True)
    shapes = []
    for date in break_up_times.index:
        shape = {"type": "line","xref": "x","yref": "paper","x0": date,"y0": 0,"x1": date,"y1": 1,"line": {"color": 'red',"width": 0.6,"dash": 'dot'},'name':'break up time'}
        shapes.append(shape)

    fig.update_layout(shapes=shapes)

    # Set title and axis properties
    fig.update_layout(
        title="Break up times & Global Variables at Tenana River-Nenana,AK",
        showlegend=True,
        xaxis=dict(range=["2010-10-01","2011-06-31"],rangeslider=dict(autorange=True),type="date"),
        yaxis=dict(anchor="x",autorange=True,domain=[0, 0.17],linecolor="black",mirror=True,range=[-55.0, 25],
                showline=True,side="left",tickfont={"color": "black"},tickmode="auto",ticks="",title="[C]",
                titlefont={"color": "black"},type="linear",zeroline=False),
        yaxis2=dict(anchor="x",autorange=True,domain=[0.17, 0.34],linecolor="black",mirror=True,range=[0, 50],
                showline=True,side="left",tickfont={"color": "black"},tickmode="auto",ticks="",title="[mm]",
                titlefont={"color": "black"},type="linear",zeroline=False),
        yaxis3=dict(anchor="x",autorange=True,domain=[0.34, 0.51],linecolor="black",mirror=True,range=[0,30],
                showline=True,side="left",tickfont={"color": "black"},tickmode="auto",ticks="",title="[m3/s]",
                titlefont={"color": "black"},type="linear",zeroline=False),
        yaxis4=dict(anchor="x",autorange=True,domain=[0.51, .68],linecolor="black",mirror=True,range=[0,200],
                showline=True,side="left",tickfont={"color": "black"},tickmode="auto",ticks="",title="[cm]",
                titlefont={"color": "black"},type="linear",zeroline=False),
        yaxis5=dict(anchor="x",autorange=True,domain=[0.68, 0.85],linecolor="black",mirror=True,range=[0,200],
                showline=True,side="left",tickfont={"color": "black"},tickmode="auto",ticks="",title="[%]",
                titlefont={"color": "black"},type="linear",zeroline=False),
        yaxis6=dict(anchor="x",autorange=True,domain=[0.85, 1],linecolor="black",mirror=True,range=[0,200],
                showline=True,side="left",tickfont={"color": "black"},tickmode="auto",ticks="",title="[W/m2]",
                titlefont={"color": "black"},type="linear",zeroline=False))


    # Update layout
    fig.update_layout(
        dragmode="zoom",hovermode="x",legend=dict(traceorder="reversed"),height=800,template="plotly",
            margin=dict(t=90,b=90),)

    fig.show()
    # Initialize figure
    fig = go.Figure()

    # Add Traces ( plots elements)
    fig.add_trace(go.Scatter(x=Data.index,y=Data["Predicted ice thickness [m]"]*100,name="Predicted ice thickness",yaxis="y4",line=dict(color='lime')))
    fig.add_trace(go.Scatter(x=Data_ice.index,y=Data_ice["IceThickness [cm]"],name="Ice thickness",yaxis="y4",line=dict(color='navy')))
    fig.add_trace(go.Scatter(x=Data.index,y=Data["Nenana: Snow depth [mm]"]/10,name='Snow depth',yaxis="y4",line=dict(color='magenta')))
    # Set title and axis properties
    fig.update_layout(
        title="Predicted ice thickness",
        showlegend=True,
        xaxis=dict(range=["1990-01-01","1995-12-31"],rangeslider=dict(autorange=True),type="date"),
        yaxis4=dict(anchor="x",autorange=True,linecolor="black",mirror=True,range=[0,200],
                showline=True,side="left",tickfont={"color": "black"},tickmode="auto",ticks="",title="[cm]",
                titlefont={"color": "black"},type="linear",zeroline=False))
    shapes = []
    for date in break_up_times.index:
        shape = {"type": "line","xref": "x","yref": "paper","x0": date,"y0": 0,"x1": date,"y1": 1,"line": {"color": 'red',"width": 0.9,"dash": 'dot'},'name':'break up time'}
        shapes.append(shape)

    fig.update_layout(shapes=shapes)
    fig.update_layout(
        dragmode="zoom",hovermode="x",legend=dict(traceorder="reversed"),height=800,template="plotly",
            margin=dict(t=90,b=150),)

    fig.show()

def standard_plot(Data,Ice):
    """
    Ice: One column with the data. We assume same length as data ( if we create ice from temperature data it should match)
    Data:Data=pd.read_csv('https://github.com/iceclassic/sandbox/blob/main/content/Time_series_DATA.txt?raw=true',index_col=0,skiprows=149)

    """
    
    # HERE WE ASSUME THAT THE ICE DATA WAS PRODUCED FROM THE TEMP DATA IN THE DF SO WE DONT WORRY ABOUT DIFFERENT LENGTHS
    # Getting the df ready to plot and fit 
    filtered_data = Data.dropna(subset=['Predicted ice thickness [m]', 'sIceThickness [cm]']).copy()
    years = filtered_data.index.year.unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
 
    drop_mask = Data[['Predicted ice thickness [m]', 'IceThickness [cm]']].notna().all(axis=1)

    filtered_data=Data[drop_mask]
    Ice=Ice[drop_mask]

    years = filtered_data.index.year.unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
    #==========================================================================
    filtered_data['Residuals'] = Ice * 100 - filtered_data['IceThickness [cm]']

    #
    fig, ax = plt.subplots(5, 1, figsize=(20, 30))

    plot_style={
                'Style1':{'color':'navy','alpha':0.8,'linestyle':'-'},
                'Style2':{'color':'magenta','alpha':0.8,'linestyle':'--'},
                'Cmap':{'cmap':'viridis'}}


    #----------PLOT 1----------------------------------------#

    ax[0].plot(filtered_data.index,Ice*100, **plot_style['Style1'], label='Predicted ice thickness')
    ax[0].plot(filtered_data.index,filtered_data["IceThickness [cm]"], **plot_style['Style2'], label='Measured ice thickness')
    ax[0].set_ylabel('Predicted Ice Thickness [cm]')
    ax[0].legend()
    ax[0].set_title('Predicted ice thickness')
    
    L2=np.sqrt(np.sum(filtered_data['Residuals'] ** 2))

    #----------PLOT 2----------------------------------------#
    sc = ax[1].scatter(filtered_data['Days until break up'], Ice * 100, c=filtered_data.index.year,**plot_style['Cmap'],s=1)
    for i, year in enumerate(years):
        year_data = filtered_data[filtered_data.index.year == year]
        year_data_ice= Ice[filtered_data.index.year == year]
        ax[1].plot(year_data['Days until break up'], year_data_ice* 100, marker=',', color=colors[i], alpha=0.8)

    cbar = plt.colorbar(sc)
    cbar.set_label('Year')

    ax[1].scatter(filtered_data['Days until break up'], filtered_data["IceThickness [cm]"], color="magenta", alpha=0.8,label='Measured ice thickness', marker='1', s=100)
    ax[1].set_ylabel('Predicted Ice Thickness [cm]')
    ax[1].set_xlabel('Days until break up')
    ax[1].legend()
    ax[1].set_title('Predicted ice thickness, normalized to days until break up')

    #----------PLOT 3----------------------------------------#
    sc = ax[2].scatter(filtered_data['Days until break up'], filtered_data['Residuals'], c=filtered_data.index.year, cmap='viridis', alpha=0.8,s=1)
    cbar = plt.colorbar(sc)
    cbar.set_label('Year')

    for i, year in enumerate(years):
        year_data = filtered_data[filtered_data.index.year == year]
        ax[2].plot(year_data['Days until break up'], year_data['Residuals'], marker='1', color=colors[i], alpha=0.8)

    # Polynomial fit
    n_degree = 1
    coefficients, residuals, _, _, _= np.polyfit(filtered_data['Days until break up'], filtered_data['Residuals'], n_degree,full=True)
    polynomial = np.poly1d(coefficients)
    x_fit = np.linspace(-130, 0, 120)
    y_fit = polynomial(x_fit)
    ax[2].plot(x_fit, y_fit, color='k', label=f'Polynomial Fit (degree={n_degree}, L2={np.sqrt(residuals[0]):.2f})', linestyle='--', linewidth=3)

    ax[2].set_ylabel('Residuals [cm]')
    ax[2].set_xlabel('Days to break up')
    ax[2].set_title('Predicted ice thickness residuals')
    ax[2].legend()


    #----------PLOT 4----------------------------------------#
    for year, color in zip(filtered_data.index.year.unique(), colors):
        year_data = filtered_data[filtered_data.index.year == year]
        ax[3].hist(year_data['Residuals'], bins=15, density=True, alpha=0.5, label=f'Year {year}', color=color)
    ax[3].set_xlabel('Residuals[cm]')
    ax[3].set_ylabel('Density')
    ax[3].set_title('Distribution of residuals per year')
    ax[3].set_xlim([-70,70])


    # Create a ScalarMappable for color mapping
    sm = ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(years), vmax=max(years)))
    sm.set_array([])  # Empty array since we just want to map colors

    # Add color bar
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Years')












    # normal fit
    aggregated_data=filtered_data['Residuals'] 
    mu, std = norm.fit(aggregated_data)
    ks_statistic, p_value = kstest(aggregated_data, 'norm', args=(mu, std))
    print("KS Statistic:", ks_statistic)
    print("P-value:", p_value)
    x = np.linspace(-70,70, 140)
    p = norm.pdf(x, mu, std)
    ax[4].plot(x, p, 'k', linewidth=2, label=f'Normal Fit ($\mu={mu:.2f}$, $\sigma={std:.2f}$)')
    ax[4].hist(aggregated_data, bins='auto',density=True, alpha=0.4,color='k', label='Aggregated ')
    ax[4].set_xlabel('Residuals[cm]')
    ax[4].set_ylabel('Density')
    ax[4].set_title('Distribution of aggregated residuals')
    ax[4].legend()
    ax[4].set_xlim([-70,70])
    # Show plot
    plt.show()
