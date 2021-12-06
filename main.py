#pip install numpy pandas (py -m pip install matplotlib plotly_express)
#pip install -U scikit-learn (streamlit streamlit-lottie streamlit-embedcode)
#pip install -U featuretools (lazypredict pycaret seaborn statsmodels)

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import streamlit as st

def stat_main():
    file_upload = st.file_uploader('Unggclickah Data')
    if file_upload is not None:
        c1, c2 = st.columns(2)
        st.success('Uji Statistik 1 Grup')
        df = pd.read_csv(file_upload)
        with c1:
            st.dataframe(df)
        with c2:
            fig, ax = plt.subplots()
            boxplot = plt.boxplot(df.iloc[:, 0])
            st.pyplot(fig)
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
            st.write('didapatkan nilai p value = {:.2f}'.format(p_value[0])
        elif df.shape[1] == 2:
            Hubungan = st.sidebar.radio()
                'Hubungan Grup', ['Berpasangan', 'Tidak Berpasangan']
            )
            stat_normal, p_value_normal = stats.shapiro(df.iloc[:, 0] - df.iloc[:, 1])
            stat_levene, p_value_levene = stats.levene(df.iloc[:, 0], df.iloc[:, 1])
            if p_value_levene > .5:
                if Hubungan == 'Berpasangan' and p_value_normal > .05:
                    statistic, p_value = stats.ttest_rel(df.iloc[:, 0], df.iloc[:, 1])
                    st.write('p value normal = {:.2f}'.format(p_value_normal))
                    st.write('dengan Uji T Test Berpasangan')
                    st.write('didapatkan nilai p value = {:.2f}'.format(p_value))
                elif Hubungan == 'Berpasangan' and p_value_normal <= .05:
                    statistic, p_value = stats.mannwhitneyu(df.iloc[:,0], df.iloc[:, 1])
                    st.write('p value normal {:.2f}'.format(p_value_normal))
                    st.write('dengan Uji Mann WhitneyU Test Berpasangan')
                    st.write('didapatkan p value = {:.2f}'.format(p_value))
                #elif Hubungan == 'Tidak Berpasangan' and 
    else:
        st.stop()
if __name__ == '__main__':
    stat_main()

