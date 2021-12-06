#pip install numpy pandas (py -m pip install matplotlib plotly_express)
#pip install -U scikit-learn (streamlit streamlit-lottie streamlit-embedcode)
#pip install -U featuretools (lazypredict pycaret seaborn statsmodels)

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
import plotly_express as px, researchpy as rp

tempat_upload = st.file_uploader('**Unggah Data**')
if tempat_upload is not None:
    df = pd.read_csv(tempat_upload)
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(df)
    with c2:
        fig = px.box(df)
        st.plotly_chart(fig)
    if df.shape[1] == 1:
        q1, q2, q3 = df.iloc[:, 0].quantile([.25, .5, .75])
        iqr = q3 - q1
        batas_bawah = q1 - (1.5*iqr)
        batas_atas = q3 + (1.5*iqr)
        popmean = st.sidebar.slider('rerata pembanding', batas_bawah, batas_atas)
        statistic, p_value = stats.ttest_1samp(df, popmean = popmean)
        st.write(f'dengan Uji One sample T test dan rerata pembangingnya {popmean}')
        st.write('didapatkan nilai p value = {:.2f}'.format(p_value[0]))
    elif df.shape[1] == 2:
        Hubungan = st.sidebar.radio(
            'Hubungan Grup', ['Berpasangan', 'Tidak Berpasangan']
        )
        stat_normal, p_value_normal = stats.shapiro(df.iloc[:, 0] - df.iloc[:, 1])
        stat_levene, p_value_levene = stats.levene(df.iloc[:, 0], df.iloc[:, 1])
        st.write(f'{p_value_levene}')
        if p_value_levene > .05:
            if p_value_normal > .05:
                if Hubungan ==  'Berpasangan':
                    statistic, p_value = rp.ttest(df.iloc[:, 0],df.iloc[:, 1], paired = True)
                    st.write('Uji T Test Berpasangan dengan P Value =  {:.2f}'.format(p_value))
                else:
                    statistic, p_value = stats.ttest_rel(df.iloc[:, 0], df.iloc[:, 1])
                    st.write('Uji T Test Tidak berpasangan dengan P value = {:.2f}'.format(p_value))
            else:
                if Hubungan == 'Berpasangan':
                    statistic, p_value = stats.wilcoxon(df.iloc[:, 0], df.iloc[:, 1])
                    st.write('Uji Wilcoxon (Berpasangan) dengan nilai  P Value = {:.2f}'.format(p_value))
                else:
                    statistic, p_value = stats.mannwhitneyu(df.iloc[:, 0], df.iloc[:, 1])
                    st.write('Uji Mann WhitneyU dengan P Value = {:.2f}'.format(p_value))
else:
    st.stop()

