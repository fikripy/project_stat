#pip install numpy pandas (py -m pip install matplotlib plotly_express)
#pip install -U scikit-learn (streamlit streamlit-lottie streamlit-embedcode)
#pip install -U featuretools (lazypredict pycaret seaborn statsmodels)

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
import plotly_express as px, researchpy as rp
from itertools import combinations as C

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
        uji_1_grup = st.sidebar.selectbox(
            'Pilih Uji', ['Uji T Satu Sampel','Uji Chi Square 1 Sampel']
        )
        if uji_1_grup == 'One Sample T Test':
            q1, q2, q3 = df.iloc[:, 0].quantile([.25, .5, .75])
            iqr = q3 - q1
            batas_bawah = q1 - (1.5*iqr)
            batas_atas = q3 + (1.5*iqr)
            popmean = st.sidebar.slider('rerata pembanding', batas_bawah, batas_atas)
            statistic, p_value = stats.ttest_1samp(df, popmean = popmean)
            st.write(f'dengan Uji One sample T test dan rerata pembangingnya {popmean}')
            st.write('didapatkan nilai p value = {:.2f}'.format(p_value[0]))
        else:
            chi_stats, p_value_chi = stats.chisquare(df.iloc[:, 0])
            st.write('Dengan Uji Chi Square didapatkan P value = {:.2f}'.format(p_value_chi))
    elif df.shape[1] == 2:
        uji_2_grup = st.sidebar.selectbox(
            'Pilih Uji 2 Grup', ['Uji T', 'Uji Chi Square']
        )
        if uji_2_grup == 'Uji T':
            Hubungan = st.sidebar.radio(
                'Hubungan Grup', ['Berpasangan', 'Tidak Berpasangan']
            )
            stat_normal, p_value_normal = stats.shapiro(df.iloc[:, 0] - df.iloc[:, 1])
            stat_levene, p_value_levene = stats.levene(df.iloc[:, 0], df.iloc[:, 1])
            st.write(f'{p_value_levene}')
            if p_value_levene > .05:
                if p_value_normal > .05:
                    if Hubungan ==  'Berpasangan':
                        statistic, p_value = stats.ttest_rel(df.iloc[:, 0],df.iloc[:, 1])
                        st.write('Uji T Berpasangan dengan P Value =  {:.2f}'.format(p_value))
                    else:
                        statistic, p_value = stats.ttest_ind(df.iloc[:, 0], df.iloc[:, 1])
                        st.write('Uji T Tidak berpasangan dengan P value = {:.2f}'.format(p_value))
                else:
                    if Hubungan == 'Berpasangan':
                        statistic, p_value = stats.wilcoxon(df.iloc[:, 0], df.iloc[:, 1])
                        st.write('Uji Wilcoxon (Berpasangan) dengan nilai  P Value = {:.2f}'.format(p_value))
                    else:
                        statistic, p_value = stats.mannwhitneyu(df.iloc[:, 0], df.iloc[:, 1])
                        st.write('Uji Mann WhitneyU dengan P Value = {:.2f}'.format(p_value))
        else:
            chi, p, dof, ex = stats.chi2_contingency(df.iloc[:, 0], df.iloc[:, 1])
            st.write('nilai p value nya = {:.2f}'.format(p))
    else:
        st.write('uji 3 grup')
        def check_p_value(df):
            cols = df.columns
            data = [df[i].values for i in cols]
            dua_grup = list(C(data, 2))
            p_value_list = [stats.shapiro(i[1] - i[0])[1] for i in dua_grup]
            p_value_arr = np.array(p_value_list)
            return data, p_value_arr
        data, p_value_arr = check_p_value(df)[0], check_p_value(df)[1]
        if (p_value_arr < .05).any():
            stats_kruskall, p_value_kruskal = stats.kruskal(*data)
            st.write('Uji Kruskal dengan P Value = {:.2f}'.format(p_value_kruskal))
        else:
            stats_anova, p_value_anova = stats.f_oneway(*data)
            st.write('Uji Anova Satu Jalur dengan P Value {:.2f}'.format(p_value_anova))
else:
    st.stop()

