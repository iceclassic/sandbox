def Filter_Numpify(df,T_0,T_f,numpyfy=False,multiyear=[]):
    """ 

    numpyfy:
    ================================
    Converts the Df to numpy:
        Df.index gets transformed to three columns [Year][Month][Day]
        Column names get deleted as the array only contains .type(float64) 

    filter
    ================================
    T_0= initial date for mask, format = ` 'yyyy-mm-dd' `
    T_f= initial date for mask, format = ` 'yyyy-mm-dd'
     
    If multiyear argument is present we filter the selected years ( only accepts list of years)

    
    """
    if multiyear:
        df = df.loc[years.isin(years_to_filter)]
    else:
        mask = df.index.to_series().between(T_0,T_f)
        df=df[mask]

    if numpyfy:
        df['Year'] = df.index.year.astype(int)
        df['Month'] =df.index.month.astype(int)
        df['Day'] = df.index.day.astype(int)
        # cols = df.columns.tolist()
        # cols = ['Year', 'Month', 'Day'] + cols[:-3]
        # df = df[cols]

       # df = df.reset_index(drop=True)
        df = df.values
        print("Dataframe has been -numpified- !!!\n The original datetime index has been converted to three columns,containing year,month and day,in position -3,-2,-1.")
    return df    