import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric

st.title(' Automated Time Series Forecasting')

"this data app uses facebook's open-source prophet library to automate"

#import data
df= st.file_uploader('Import the time series csv file here')

if df is not None:
    data=pd.read_csv(df)
    data['ds']=pd.to_datetime(data['ds'], errors='coerce')

    st.write(data)

    max_date= data['ds'].max()
    #st.write(max_date)

#select the forecast horizone

periods_input= st.number_input('How many periods would you like to forecast into future?',
min_value= 1, max_value=365)

if df is not None:
    m=Prophet()
    m.fit(data)

