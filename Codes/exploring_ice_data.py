"""This module provides functions for manipulating, plotting and working with  the time-series data for selected for the Nenana Ice Classic.
This function are though on the basis that the dataframe has the same structure and it has been loaded as the README suggest.
"""
# libraries

import matplotlib.pyplot as plt

def explore_dataframe(df,plot_variables=True):
    """ 
    This functions has the purpose to get a broad understanding of the contents of the dataframe

    Inputs:
            df: dataframe
    Outputs:

            rellenar
    """

    ## Built-in methods
    df.info()
    # - from here we can check for correct 
            #- Correct index
            #- correct index tye (datetime)
            # correct column names
            # 
    if plot_variables:
        explore_data(df)


def explore_data(Data):
    for i, col in enumerate(Data.columns):
        col_data = Data[col].copy()
        col_data.dropna(inplace=True)
        if not col_data.empty: 
            plt.figure(figsize=(20, 3))
            plt.plot(col_data.index, col_data.values, label=col, color=plt.cm.tab10(i % 10))
            plt.legend()
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.title(col)
            plt.tight_layout()
            plt.show()