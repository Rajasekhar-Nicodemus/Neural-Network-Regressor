import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


app_mode = st.sidebar.selectbox('Select Page',['Date Pre-Processing','Training','Prediction']) 

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

if app_mode=='Date Pre-Processing':
    st.header("Neural Network Regressor APP: By E.Rajasekhar Nicodemus (rajasekhar.nicodemus@gmail.com)")
    dat=st.file_uploader('Upload input data in csv format:')
    st.markdown("Rules for csv file:")
    st.markdown("i) First row is header row")
    st.markdown("ii) Last column is the output")
    if dat is not None:
        
        data=pd.read_csv(dat,header=0)
        st.write(data.head(10))
        st.subheader("Size of data is  :"+str(data.shape))
        st.subheader("Histogram and Density Plots")
        st.markdown("Displays distribution of variable values") 
        nrow,ncol=data.shape
        col=data.columns
        cou=1
        
        for i in col:
            plt.rcParams.update({'font.size': 6})
            fig = plt.figure(figsize=(3, 1))
            ax = fig.add_subplot()
            l=str(data[i].name)
            ax.set_title(l)
            sns.distplot(data[i],color="g")
            st.pyplot(fig) 
            
        st.subheader("Correlation map") 
        st.markdown("Displays linear correlation between variables")  
        correlations=data.corr()
        fig_1=plt.figure(figsize=(4, 4))
        ax=fig_1.add_subplot(111)
        cax=ax.matshow(correlations,vmin=-1,vmax=1, cmap="gist_rainbow")
        fig_1.colorbar(cax)
        plt.show()
        st.pyplot(fig_1) 
        
        var=st.radio('Select Data transfomration method',['Min-Max','Standardization'])
        st.markdown("Data need to be transformed for Neural Network")
        if st.button("Transform data"):
            if(var=='Min-Max'):
                 data_norm=(data-np.min(data))/(np.max(data)-np.min(data))
                 st.write(data_norm.head(10))
                 df_norm=pd.concat([np.min(data),np.max(data)],axis=1)
                 df_norm.columns=['Min','Max']
            if(var=='Standardization'):
                 data_norm=(data-np.mean(data))/(np.std(data))
                 st.write(data_norm.head(10))
                 df_norm=pd.concat([np.mean(data),np.std(data)],axis=1)
                 df_norm.columns=['Mean','Std']
            
            
            csv=convert_df(df_norm) 
            st.download_button(label="Download Transformation Paramter file",data=csv,file_name='Transform_param.csv',mime='text/csv', )
            
    
    


        

        
