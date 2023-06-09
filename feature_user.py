import time
import os
import pandas as pd
import seaborn as sns
import numpy as np
import pickle
import sklearn.metrics as metrics
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, Input, Model
from sklearn.metrics import accuracy_score,precision_score, recall_score
from keras.layers import Input, Dense, Reshape, Flatten, Lambda
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Lambda, Layer
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from IPython.display import display
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sdv.single_table import CopulaGANSynthesizer


def vae_cvae_synthetic_generation(df,categorical_columns,condition_columns,lr_rate,latent_dim,epochs,batch_size):
    

    # categorical_columns = ['Hour of the day', 'Week of the day', 'Source', 'Destination']
    # categorical_columns=select_catagorical_column

    one_hot_encoder = OneHotEncoder(sparse=False)
    encoded_features = one_hot_encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out(categorical_columns))
    # st.write('Hey')

# Preprocess condition data: Audio Source, Audio Source Type, Genre
    # condition_columns = ['Audio Source Mode', 'Audio Channel/Stations','Genre']
    condition_encoder = OneHotEncoder(sparse=False)
    condition_features = condition_encoder.fit_transform(df[condition_columns])
    condition_data = pd.DataFrame(condition_features, columns=condition_encoder.get_feature_names_out(condition_columns))

# Concatenate encoded features with condition data
    features = pd.concat([encoded_df, condition_data], axis=1)

    input_dim = features.shape[1]
    # latent_dim = 8
    # latent_dim=latent_dims
    condition_dim = condition_data.shape[1]

    encoder_input = Input(shape=(input_dim,))

    conditional_input = Input(shape=(condition_dim,))

    encoder_combined_input = tf.keras.layers.Concatenate()([encoder_input, conditional_input])

    x = Dense(64, activation='relu')(encoder_combined_input)

    x = Dense(128, activation='relu')(x)

    new_shape = K.int_shape(x)

    z_mean = Dense(latent_dim, name='z_mean')(x) # mean values of encoded input

    z_log_var = Dense(latent_dim, name='z_log_var')(x) # Std dev(variance) of encoded input

############### REPARAMETERIZATION TRICK ##################
# z is the lambda custom layer we are adding for gradient descent calculations
# using mean and variance

    def sampling(args):
        
    
        z_mean, z_log_var = args
    
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0.0, stddev=1.0)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

# this the last layer in the encoder model
    z = Lambda(sampling, output_shape=(latent_dim, ), name='z')([z_mean, z_log_var])

####### Encoder Model #######
    encoder = Model([encoder_input,conditional_input], [z_mean,z_log_var,z], name='encoder')


    decoder_input = Input(shape=(latent_dim,))

    conditional_input_dec = Input(shape=(condition_dim,))  # Input for conditional variables

    decoder_combined_input = tf.keras.layers.Concatenate()([decoder_input, conditional_input_dec])

    x = Dense(new_shape[1], activation='relu')(decoder_combined_input)

    decoder_output = Dense(input_dim, activation='sigmoid')(x)

    decoder = Model([decoder_input, conditional_input_dec], decoder_output, name='decoder')

# combine encoder and decoder
    z = [encoder([encoder_input, conditional_input])[2], conditional_input]
    z_decoded = decoder(z)
#define the model
    cvae = Model([encoder_input,conditional_input], z_decoded, name ="cvae")

# Compute the VAE loss
    reconstruction_loss = binary_crossentropy(encoder_input,z_decoded)
    reconstruction_loss *= input_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    cvae_total_loss = K.mean(reconstruction_loss + kl_loss)

# Compile the CVAE model
    cvae.add_loss(cvae_total_loss)
    cvae.compile(optimizer=Adam(learning_rate=lr_rate))
    cvae.fit([features, condition_data], epochs = epochs, batch_size = batch_size)
    
    return one_hot_encoder,condition_encoder,df,features,condition_features,categorical_columns,condition_columns,encoder,decoder,latent_dim,condition_data,encoded_features
# data=vae_synthetic_generation(dataframe)

def generate_synthetic_data(one_hot_encoder,condition_encoder,n_samples,df,features,condition_features,categorical_columns,condition_columns,encoder,decoder,latent_dim,condition_data,encoded_features):
    synthetic= pd.DataFrame()
    for vin in (df['VIN'].unique()):
        synthetic_data = pd.DataFrame()
        encoded_vin_data_feat = features[df['VIN'] == vin]
        encoded_vin_data_label = condition_features[df['VIN'] == vin]
        z_mean_,z_log_var_, encoded_vin_latent = encoder.predict([encoded_vin_data_feat,encoded_vin_data_label])  
        random_latent_points=tf.random.normal(shape=(n_samples, latent_dim)) 
        new_latents = encoded_vin_latent.mean(axis=0) + random_latent_points * encoded_vin_latent.std(axis=0)
        random_condition_data=condition_data.sample(n=n_samples, replace=True).values
        synthetic_data = decoder.predict([new_latents, random_condition_data])
        decoded_features = one_hot_encoder.inverse_transform(synthetic_data[:, :encoded_features.shape[1]])
        decoded_condition = condition_encoder.inverse_transform(synthetic_data[:, encoded_features.shape[1]:])
        synthetic_df = pd.DataFrame(decoded_features, columns=categorical_columns)
        synthetic_condition_df = pd.DataFrame(decoded_condition, columns=condition_columns)
        synthetic_data = pd.concat([synthetic_df, synthetic_condition_df], axis=1)
        synthetic_data['VIN'] = vin
        # synthetic=synthetic.append(synthetic_data) 
        synthetic=pd.concat([synthetic,synthetic_data])
    synthetic = synthetic[df.columns]
    synthetic = pd.concat([synthetic,df],ignore_index=True)

    return synthetic

#*********************************************************************************************************************************
def vae_generated_synthetic_data(df,categorical_columns,lr_rate,latent_dim,epochs,batch_size):
    # categorical_cols = [ 'Hour of the day', 'Week of the day', 'Audio Source Mode',
    #    'Audio Channel/Stations', 'Genre','Source','Destination']
    enc = OneHotEncoder(sparse=False)
    enc.fit(df[categorical_columns])
    encoded_cols = enc.transform(df[categorical_columns])

    # Define the VAE model

    input_dim = encoded_cols.shape[1]
    # latent_dim = 8

    encoder_input = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(encoder_input)
    x = Dense(128, activation='relu')(x)
    new_shape = K.int_shape(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),mean=0.0, stddev=1.0)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim, ), name='z')([z_mean, z_log_var])

    encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')

    decoder_input = Input(shape=(latent_dim,))
    x = Dense(new_shape[1], activation='relu')(decoder_input)
    x = Dense(64, activation='relu')(x)
    decoder_output = Dense(input_dim, activation='sigmoid')(x)
    decoder = Model(decoder_input, decoder_output, name='decoder')

    vae_output = decoder(encoder(encoder_input)[2])
    vae = Model(encoder_input, vae_output, name='vae')

    # Compute the VAE loss
    reconstruction_loss = binary_crossentropy(encoder_input,vae_output)
    reconstruction_loss *= input_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_total_loss = K.mean(reconstruction_loss + kl_loss)

    # Compile the CVAE model
    vae.add_loss(vae_total_loss)
    vae.compile(optimizer=Adam(learning_rate=lr_rate))
    start_time = time.time()
    for epoch in tqdm(range(100)):
        vae.fit(encoded_cols, epochs=epochs, batch_size=batch_size, verbose=0)
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")
    return enc,encoded_cols,df,categorical_columns,encoder,decoder,latent_dim


def generate_synthetic_data_vae(enc,encoded_cols,n_samples,df,categorical_columns,encoder,decoder,latent_dim):
            synthetic_data = pd.DataFrame()          
            for vin in df['VIN'].unique():
                    synthetic= pd.DataFrame()
                    encoded_vin_data = encoded_cols[df['VIN'] == vin]
                    _, _, encoded_vin_latent = encoder.predict(encoded_vin_data)

                    # Generate new latent samples

                    latent_samples = np.random.normal(size=(n_samples, latent_dim))
                    new_latents = encoded_vin_latent.mean(axis=0) + latent_samples * encoded_vin_latent.std(axis=0)

                    # Decode the new latent samples into new samples

                    new_samples = decoder.predict(new_latents)

                    # Add the new samples to the synthetic data
                    decoded_samples = enc.inverse_transform(new_samples)
                    synthetic=pd.DataFrame(decoded_samples,columns=categorical_columns)
                    synthetic['VIN']=vin
                    synthetic_data=pd.concat([synthetic_data,synthetic])
            return synthetic_data

# synthetic_data = generate_synthetic_data(50000)

def copulagan(df,synthesizer,n_samples):
    
    synthesizer.fit(df)

    synthetic_data = pd.DataFrame()
    for vin in (df['VIN'].unique()):
        
        synthetic = synthesizer.sample(num_rows=n_samples)
        
        synthetic['VIN'] = vin
        synthetic_data = pd.concat([synthetic_data,synthetic])
        
    # Combine the synthetic data into a single DataFrame
    synthetic_data = pd.concat([synthetic_data,df])

    return synthetic_data

def fast_ml(df,synthesizer,n_samples):
    synthesizer.fit(df)

    synthetic_data = pd.DataFrame()
    for vin in (df['VIN'].unique()):
        
        synthetic = synthesizer.sample(num_rows=n_samples)
        
        synthetic['VIN'] = vin
        synthetic_data = pd.concat([synthetic_data,synthetic])
        
    # Combine the synthetic data into a single DataFrame
    synthetic_data = pd.concat([synthetic_data,df])

    return synthetic_data

def gaussian_copula(df,synthesizer,n_samples):
    synthesizer.fit(df)

    synthetic_data = pd.DataFrame()
    for vin in (df['VIN'].unique()):
        
        synthetic = synthesizer.sample(num_rows=n_samples)
        
        synthetic['VIN'] = vin
        synthetic_data = pd.concat([synthetic_data,synthetic])
        
    # Combine the synthetic data into a single DataFrame
    synthetic_data = pd.concat([synthetic_data,df])

    return synthetic_data

def ctgan(df,synthesizer,n_samples):
    synthesizer.fit(df)

    synthetic_data = pd.DataFrame()
    for vin in (df['VIN'].unique()):
        
        synthetic = synthesizer.sample(num_rows=n_samples)
        
        synthetic['VIN'] = vin
        synthetic_data = pd.concat([synthetic_data,synthetic])
        
    # Combine the synthetic data into a single DataFrame
    synthetic_data = pd.concat([synthetic_data,df])

    return synthetic_data


def tvae(df,synthesizer,n_samples):
    synthesizer.fit(df)

    synthetic_data = pd.DataFrame()
    for vin in (df['VIN'].unique()):
        
        synthetic = synthesizer.sample(num_rows=n_samples)
        
        synthetic['VIN'] = vin
        synthetic_data = pd.concat([synthetic_data,synthetic])
        
    # Combine the synthetic data into a single DataFrame
    synthetic_data = pd.concat([synthetic_data,df])

    return synthetic_data


