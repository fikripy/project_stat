#pip install numpy pandas (py -m pip install matplotlib plotly_express)
#pip install -U scikit-learn (streamlit streamlit-lottie streamlit-embedcode)
#pip install -U featuretools (lazypredict pycaret seaborn statsmodels)

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
import plotly_express as px, researchpy as rp
from itertools import combinations as C
from statsmodels.stats.contingency_tables import mcnemar

tempat_upload = st.file_uploader('**Unggah Data**')
if tempat_upload is not None:
    df = pd.read_csv(tempat_upload)
    df_ori = df.copy()
#Mengecek Homogenitas Data
    def periksa_homogenitas(df):
        df_copy = df.copy()
        if df.shape[1] > 1:
            cols = df.columns
            data = [df[i].values for i in cols]
            p_levene = stats.levene(*data)[1]
            if p_levene < .05:
                #print('log trans')
                df = np.log(df + (-df.min().min()) + 1)
                data_trans = [df[i].values for i in cols]
                p_levene2 = stats.levene(*data_trans)[1]
                if p_levene2 < .05:
                    #print('sqrt trans')
                    df = np.sqrt(df_copy + (-df_copy.min().min()) + 1)
                else: pass
            else: pass
        else: pass
        return df
    df = periksa_homogenitas(df)
    def display(df):
        if df.equals(df_ori):
            c1, c2 = st.columns(2)
            with c1:
                st.dataframe(df_ori)
            with c2:
                fig = px.box(df_ori)
                st.plotly_chart(fig)
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.dataframe(df_ori)
            with c2:
                st.dataframe(df)
            with c3:
                fig = px.box(df)
                st.plotly_chart(fig)
    display(df)
    if df.shape[1] == 1:
        uji_1_grup = st.sidebar.selectbox(
            'Pilih Uji', ['Uji T Satu Sampel','Uji Chi Square 1 Sampel']
        )
        if uji_1_grup == 'Uji T Satu Sampel':
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
        #t_or_aov = st.sidebar.selectbox('Jumlah Grup', ['2', '>2'])
        uji_2_grup = st.sidebar.selectbox(
            'Pilih Uji 2 Grup', ['Uji T', 'Uji Chi Square', '>2 grup']
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
        uji_3_grup = st.sidebar.selectbox(
            'Uji Statistik', ['Anova','Chi Squared']
        )
        def check_p_value(df):
            cols = df.columns
            data = [df[i].values for i in cols]
            dua_grup = list(C(data, 2))
            p_value_list = [stats.shapiro(i[1] - i[0])[1] for i in dua_grup]
            p_value_arr = np.array(p_value_list)
            return data, p_value_arr
        data, p_value_arr = check_p_value(df)[0], check_p_value(df)[1]
        if uji_3_grup == 'Anova':
            Hubungan = st.sidebar.selectbox(
                'Hubungan', ['Berpasangan', 'Tidak Berpasangan']
            )
            if (p_value_arr < .05).any():
                if Hubungan == 'Tidak Berpasangan':
                    stats_kruskall, p_value_kruskal = stats.kruskal(*data)
                    st.write('Uji Kruskal dengan P Value = {:.2f}'.format(p_value_kruskal))
                else:
                    stats_friedman, p_value_friedman = stats.friedmanchisquare(*data)
                    st.write('Uji Friedman dengan P value = {:.2f}'.format(p_value_friedman))
            else:
                stats_anova, p_value_anova = stats.f_oneway(*data)
                st.write('Uji Anova Satu Jalur dengan P Value {:.2f}'.format(p_value_anova))
                
        else:
            chi, p, dof, ex = stats.chi2_contingency(*data)
            st.write('nilai p value nya = {:.2f}'.format(p))

else:
    st.stop()

