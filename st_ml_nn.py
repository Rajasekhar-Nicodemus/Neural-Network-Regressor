import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sklearn
from sklearn.model_selection import train_test_split
import graphviz





app_mode = st.sidebar.selectbox('Select Page',['Data Pre-Processing','Training','Prediction']) 
st.sidebar.graphviz_chart('''
                          digraph  {
                           
		P [label="Data Pre-Processing" color=Blue, fontcolor=Red, fontsize=10, shape=box] 
        a [label="Transformed data file" color=black, fontcolor=darkgreen, fontsize=10,shape=egg] 
		T [label="Training" color=Blue, fontcolor=Red, fontsize=10, shape=box] 
        b[label="Transformation paramter file" color=black, fontcolor=darkgreen, fontsize=10] 
        C [label="Prediction" color=Blue, fontcolor=Red, fontsize=10, shape=box] 
        d [label="NN parmeter file and Weights file" color=black, fontcolor=darkgreen, fontsize=10, shape=egg] 
        
		P->a
        a->T
        T->d
        d->C
        P->b
        b->C
    }                                  
                          ''')

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv(index=False,header=False).encode('utf-8')
 
@st.cache
def convertx_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv(index=False,header=True).encode('utf-8')
 
@st.cache
def relu(x):
    return(np.maximum(0,x))

@st.cache
def sigmoid(x):
    return(1/(1 + np.exp(-x)))
 
data_pre=[]

if app_mode=='Data Pre-Processing':
    st.header("Neural Network Regressor APP: By E.Rajasekhar Nicodemus (rajasekhar.nicodemus@gmail.com)")
    st.markdown("Prefered usage on laptop/workstation for better experience")
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
            
            
            csv=convertx_df(df_norm) 
            csv_data=convertx_df(data_norm)
            st.download_button(label="Download Transformation Paramter file (For Prediction Section)",data=csv,file_name='Transform_param.csv',mime='text/csv', )
            st.download_button(label="Transformed Data for Training (Training Section)",data=csv_data,file_name='Transformed_data.csv',mime='text/csv', )
    
if app_mode=='Training': 
    st.header("Neural Network Regressor APP: By E.Rajasekhar Nicodemus (rajasekhar.nicodemus@gmail.com)")
    st.markdown("Prefered usage on laptop/workstation for better experience")
    dat=st.file_uploader('Upload Transformed data from Data Pre-Processing Section')
    if dat is not None:
       data_norm=pd.read_csv(dat,header=0)
       st.write(data_norm.head(10))
       st.subheader("Size of data is  :"+str(data_norm.shape))
       x=data_norm.iloc[:,:data_norm.shape[1]-1]
       y=data_norm[data_norm.columns[len(data_norm.columns)-1]]
       
       st.header("Test-Train Split")
       r=st.number_input("Enter test total data ratio",0.10,1.00)
       X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=r, random_state=42)
       st.header("Train  data")
       st.write(X_train.head(10))
       st.header("Test  data")
       st.write(X_test.head(10))
       csv=convert_df(X_train) 
       st.download_button(label="Download X-train data",data=csv,file_name='X_train.csv',mime='text/csv', )
       csv=convert_df(X_test) 
       st.download_button(label="Download X-test",data=csv,file_name='X_test.csv',mime='text/csv', )
       csv=convert_df(y_train) 
       st.download_button(label="Download y-train data",data=csv,file_name='y_train.csv',mime='text/csv', )
       csv=convert_df(y_test) 
       st.download_button(label="Download y-test",data=csv,file_name='y_test.csv',mime='text/csv', )
       
       
       
       st.header("Neural Network Model Architecture")
       st.markdown("![Alt Text](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/375px-Colored_neural_network.svg.png)")
       st.markdown("Image sorce: Wikipedia")
       neurnons=[]
       n_layers=st.slider('Select number of layers:', 2,7)
       st.write(n_layers)
       for i in range(0,n_layers):
           n=st.slider('Select number of neurons in layer'+str(i+1), 5,25)
           neurnons.append(n)
           st.write(neurnons[i]) 
       actv=st.selectbox('Select activation function',['sigmoid','tanh','relu'])   
       
       
       nn_param=[[actv],[n_layers],[data_norm.shape[1]-1],neurnons]
       frame_nn = pd.DataFrame(nn_param)
       #st.write(frame_nn.iloc[1,0])
  

       model = Sequential()
       for i in range(0,n_layers+1):
           if(i==0):
            model.add(Dense(neurnons[i], input_shape=(data_norm.shape[1]-1,), activation=actv)) 
           elif i==n_layers:
             model.add(Dense(1,activation='linear'))  
           else:
            model.add(Dense(neurnons[i], activation=actv))
            
            
       if st.button("Model Summary"):
           model.summary(print_fn=lambda x: st.text(x))
           
   
       model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss='mse',metrics=['mse'])  
       
       csv=convert_df(frame_nn) 
       st.download_button(label="Download NN Parametrs (For Prediction Section)",data=csv,file_name='nn_pram.csv',mime='text/csv', )
       
       
       st.header("Training Parmeters")
       epoch=st.number_input("Enter number of epochs",100,10000)
       batch=st.number_input("Enter batch size",1,data_norm.shape[0])
       val_split=st.number_input("Enter validation split",0.01,0.50)
       lr=st.number_input("Enter learning rate",1e-8,0.1,format="%.10f")
       model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss='mse',metrics=['mse'])  
       st.markdown("MSE values may look low as NN is being devloped on transomfed data. Please double check model for sufficent accuarcy and sufficiently low MSE")
       val=[]
       train=[]
       
       test_mse=0
       
       if st.button("Train"):
           model.summary(print_fn=lambda x: st.text(x))
           tf.keras.backend.clear_session()
           my_bar = st.progress(0)
           p = st.empty()
           p1=st.empty()
           for i in range(0,epoch):
             history=model.fit(X_train, y_train, epochs=1, batch_size=batch, validation_split=val_split, verbose=1)
             my_bar.progress((i+1)/epoch)
             p.write("Epoch :"+str(i+1)+"/"+str(epoch))
             losses = history.history['mse']
             losses_v = history.history['val_mse']
             p1.write("Train mse :"+str(losses)+" Val mse :"+str(losses_v))
             train.append(losses)
             val.append(losses_v)
           fig = plt.figure(figsize=(6, 5))
           ax = fig.add_subplot()
           plt.plot(train)
           plt.plot(val)
           plt.title('Model mse')
           plt.ylabel('loss')
           plt.xlabel('epoch')
           ax.legend(['train','val'], loc='upper left')    
           st.pyplot(fig)  
           y_pred=model.predict(X_test)
           r2=sklearn.metrics.r2_score(y_test,y_pred) 
           test_mse=sklearn.metrics.mean_squared_error(y_test,y_pred) 
           st.write("Mse of test data: "+str(test_mse))
           if r2>0:
             st.write("R2 for test data :"+ str(r2))
           
           weights_a=model.layers[0].get_weights()[0]
           weights_b=model.layers[0].get_weights()[1]

           weights=pd.DataFrame(weights_a)
           weight=pd.DataFrame(weights_b)
           frames=[weights,weight]
           weights_join=pd.concat(frames,axis=0)  
           
           for i in range(1,n_layers+1):  
             weights_a=model.layers[i].get_weights()[0]
             weights_b=model.layers[i].get_weights()[1]

             weights=pd.DataFrame(weights_a)
             weight=pd.DataFrame(weights_b)
             frames=[weights_join,weights]
             weights_join=pd.concat(frames,axis=0)
             frames=[weights_join,weight]
             weights_join=pd.concat(frames,axis=0)
           
           #st.write(weights_join)
           csv=convert_df(weights_join) 
           st.download_button(label="Download weights (For Prediction Section)",data=csv,file_name='weights.csv',mime='text/csv', )
           x_check=x[0:8]
           y_check=y[0:8]
           pred_check=model.predict(x_check)
           #st.write(x_check)
           #st.write(y_check)
           #st.write(pred_check)
           
           
if app_mode=='Prediction': 
    st.header("Neural Network Regressor APP: By E.Rajasekhar Nicodemus (rajasekhar.nicodemus@gmail.com)")
    st.markdown("Prefered usage on laptop/workstation for better experience")
    dat=st.file_uploader('Upload data in csv format for prediction:')
    st.markdown("Rules for csv file:")
    st.markdown("i) Must contain only inputs and not ouput")
    st.markdown("ii) One less column than the cs vile in Dat Pre-processing section")
    if dat is not None:   
        data=pd.read_csv(dat,header=0)
        st.write(data.head(3))    
        dat_param=st.file_uploader('Upload parameter file for transformation (Can be downloaded from Data Pre-Processing section) :')
        if dat_param is not None:   
           data_param=pd.read_csv(dat_param,header=0)
           #st.write(data_param)
           nrow,rcol=data_param.shape
           trans_input=data_param.iloc[:nrow-1,:]
           trans_output=data_param.iloc[nrow-1:nrow,:]
           nrow_i,ncol_i=trans_input.shape
           
           if(nrow_i==data.shape[1]):
               st.success("Files are appropraite")
           else:
               st.error("Files are incompatible, Please reupload- files")
         
           if(nrow_i==data.shape[1]):
               data_trans=None
               if(data_param.columns[0]=='Min'):
                   c=data.to_numpy()
                   a=trans_input['Min'].to_numpy()
                   b=trans_input['Max'].to_numpy()
                   data_trans=(c-np.transpose(a))/(np.transpose(b)-np.transpose(a))
                   st.write("Normalized Data")
                   st.write(data_trans)
               elif(data_param.columns[0]=='Mean'):
                   c=data.to_numpy()
                   a=trans_input['Mean'].to_numpy()
                   b=trans_input['Std'].to_numpy()
                   data_trans=(c-np.transpose(a))/(np.transpose(b))
                   st.write("Normalized Data")
                   st.write(data_trans)
               else:
                   st.error("Check transfomration parameter file")
               
               if data_trans is not None:
                   nn_param=st.file_uploader('Upload NN paramter file: (From Training Section)')
                   if nn_param is not None:
                     nn_param_data=pd.read_csv(nn_param,header=None)
                     #st.write( nn_param_data)
                     #st.write(nrow_i)
                     if(int(nn_param_data.iloc[2][0])==nrow_i):
                           st.success("Files are appropraite")
                     else:
                          st.error("Files are incompatible, Please reupload- files")
 
                     if(int(nn_param_data.iloc[2][0])==nrow_i):  
                           lay=[]
                           for i in range(0,int(nn_param_data.iloc[1][0])):
                               lay.append(int(nn_param_data.iloc[3][i]))
                          #st.write("Number of Neurons in each layer")   
                          #st.write(lay)
                          
                           weight_param=st.file_uploader('Upload weights (Can be downloaded from Training section) :')
                           if weight_param is not None:
                              weights=pd.read_csv(weight_param,header=None)
                              #st.write(weights)
                              act=nn_param_data.iloc[0][0]
                              
                              for i in range(0,int(nn_param_data.iloc[1][0])+1):
                                  
                                  if(i==0):
                                      w=weights.iloc[0:nrow_i,0:lay[i]].to_numpy()
                                      r=nrow_i
                                      b=weights.iloc[r:r+lay[i]][0].to_numpy()
                                      r=r+lay[i]
                                      #st.write(w)
                                      #st.write(b)
                                      if(act=="tanh"):
                                          out=np.tanh(np.matmul(data_trans,w)+np.tile(b.reshape(lay[i]),(data.shape[0],1)))
                                      if(act=="sigmoid"):
                                          out=sigmoid(np.matmul(data_trans,w)+np.tile(b.reshape(lay[i]),(data.shape[0],1)))
                                      if(act=="relu"):
                                          out=relu(np.matmul(data_trans,w)+np.tile(b.reshape(lay[i]),(data.shape[0],1)))
                                      
                                      
                                      #st.write(out.shape)
                                      #st.write(np.tile(b.reshape(lay[i]),(data.shape[0],1)))
                                      #st.write("check")
                                      
                                  elif(i==int(nn_param_data.iloc[1][0])):  
                                      w=weights.iloc[r:r+lay[i-1],0:1].to_numpy()
                                      r=r+int(lay[i-1])
                                      b=weights.iloc[r:r+1][0].to_numpy()
                                      #st.write(w)
                                      #st.write(b)
                                      if(act=="tanh"):
                                          out=np.matmul(out,w)+np.tile(b.reshape(1),(data.shape[0],1))
                                      if(act=="sigmoid"):
                                          out=np.matmul(out,w)+np.tile(b.reshape(1),(data.shape[0],1))
                                      if(act=="relu"):
                                          out=np.matmul(out,w)+np.tile(b.reshape(1),(data.shape[0],1)) 
                                      
                                      #st.write(out.shape)
                                      #st.write(out)
                                   
                                   
                                  else:
                                      w=weights.iloc[r:r+lay[i-1],0:lay[i]].to_numpy()
                                      r=r+int(lay[i-1])
                                      b=weights.iloc[r:r+int(lay[i])][0].to_numpy()
                                      r=r+int(lay[i])
                                      #st.write(w)
                                      #st.write(b)
                                      
                                      if(act=="tanh"):
                                          out=np.tanh(np.matmul(out,w)+np.tile(b.reshape(lay[i]),(data.shape[0],1)))
                                      if(act=="sigmoid"):
                                         out=sigmoid(np.matmul(out,w)+np.tile(b.reshape(lay[i]),(data.shape[0],1)))
                                      if(act=="relu"):
                                          out=relu(np.matmul(out,w)+np.tile(b.reshape(lay[i]),(data.shape[0],1)))
                                     
                                      
                                      #st.write(out.shape)
                                      
                                      
                              if(data_param.columns[0]=='Min'):
                               
                                  a=trans_output['Min'].to_numpy()
                                  b=trans_output['Max'].to_numpy()
                                  out_predict=(out)*(np.transpose(b)-np.transpose(a))+np.transpose(a)
             
                              elif(data_param.columns[0]=='Mean'):
                            
                                  a=trans_output['Mean'].to_numpy()
                                  b=trans_output['Std'].to_numpy()
                                  out_predict=(out)*(np.transpose(b))+np.transpose(a)
                                  
                              if st.button("Predict"):
                                 st.write(out_predict)
               
        
 
                
                          
        
