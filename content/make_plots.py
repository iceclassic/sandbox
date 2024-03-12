
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
