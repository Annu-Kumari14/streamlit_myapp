import streamlit as st
from PIL import Image
import pandas as pd
from io import StringIO
import time
import base64
from xgboost import XGBClassifier
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from feature_user import vae_cvae_synthetic_generation, generate_synthetic_data,vae_generated_synthetic_data,generate_synthetic_data_vae,copulagan,fast_ml,gaussian_copula,ctgan,tvae
import streamlit.components.v1 as components
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
from sdv.single_table import CopulaGANSynthesizer
from sdv.lite import SingleTablePreset
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdmetrics.reports.single_table import QualityReport
import json
from functools import reduce



# Define the pages
pages = {
    "About the App":"Providing an info to the user about the app",
    "Upload-generate-check-quality-score": "Upload-generate-check-quality-score"
    # "Validation": "Validate Synthetic Data",
    # "Visualization": "Visualize your data"
}


# Define the current page
current_page = st.sidebar.radio("Navigation", list(pages.keys()))

if current_page == "About the App":
    st.title("Hey user!")
    st.header("A walkthrough to follow the steps:")
    # col1, col2, col3 = st.columns(3)
    tab1, tab2, tab3 = st.tabs(["Step 1", "Step 2", "Step 3"])

    with tab1:

        st.subheader("Upload your real data")
        st.image("data.jpg")
        expander = st.expander("Click Me for more info")
        expander.write("Hi User! This app will help you to generate synthetic data with different synthesizers. Just be ready with your real data!")

    with tab2:
        st.subheader("Generate the synthetic data")
        st.image("gd.png")
        expander=st.expander("Click Me")
        expander.write("When you click the button Generate and save synthetic data,it will basically generate all type of synthetic data based on different synthesizers and save the data on your local machine.")
        expander.write("Conditional Variational Autoencoders (CVAE) is use to generate synthetic data samples. Refer the link below to read about it more.")
        expander.markdown("https://towardsdatascience.com/understanding-conditional-variational-autoencoders-cd62b4f57bf8")
        expander.write("A Variational Autoencoder (VAE) is a deep learning model that can generate new data samples.")
        expander.markdown("https://www.scaler.com/topics/deep-learning/variational-autoencoder/")
        expander.image("vae.png")
        expander.write("Now let's introduce some synthesizers which we have used for generation of synthetic data:")

        expander.write("The Copula GAN Synthesizer uses a mix classic, statistical methods and GAN-based deep learning methods to train a model and generate synthetic data.")
        expander.markdown("https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/copulagansynthesizer")
        expander.write("The Fast ML Preset synthesizer is optimized for modeling speed.")
        expander.markdown("https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/fast-ml-preset")
        expander.write("The Gaussian Copula Synthesizer uses classic, statistical methods to train a model and generate synthetic data.")
        expander.markdown("https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/gaussiancopulasynthesizer")
        expander.write("The CTGAN Synthesizer uses GAN-based, deep learning methods to train a model and generate synthetic data.")
        expander.markdown("https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/ctgansynthesizer")
        expander.write("The TVAE Synthesizer uses a variational autoencoder (VAE)-based, neural network techniques to train a model and generate synthetic data.")
        expander.markdown("https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/tvaesynthesizer")

    with tab3:
        st.subheader("Check Quality Report")
        st.image("score.jpg")
        expander=st.expander("Click Me")
        expander.write("Quality report evaluates the shapes of the columns (marginal distributions) and the pairwise trends between the columns (correlations). Refer the below link to read about it more")
        expander.markdown("https://docs.sdv.dev/sdmetrics/reports/quality-report/whats-included")
    # expander = st.expander("Click Me for more info")
    # expander.write(
    #     "Hi User! This app will help you to generate synthetic data with different synthesizers. Just be ready with your real data!"
        
    # )
    


# Render the content based on the current page
elif current_page == "Upload-generate-check-quality-score":
    st.title("Welcome to the world of generating synthetic data!")
    image = Image.open('header.png')
    st.image(image, caption='Synthetic Data Generation')
    uploaded_file = st.file_uploader("Please upload your real data")
    if uploaded_file is not None:
        st.success('Data uploaded Successfully!', icon="✅")
    # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)

    # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # st.write(stringio)

    # To read file as string:
        string_data = stringio.read()
    # st.write(string_data)
   
        dataframe= pd.read_csv(uploaded_file)
        st.write(dataframe)
        select_input_column = st.multiselect('Select input columns for CVAE',options=list(dataframe.columns),key="1")
        st.write(select_input_column)
        select_conditional_column = st.multiselect('Select conditional columns for CVAE',options=list(dataframe.columns))
        st.write(select_conditional_column)
        select_input_column_vae = st.multiselect('Select input columns for VAE',options=list(dataframe.columns),key="2")
        st.write(select_input_column_vae)
        st.info('Primary key should be the unique identified column such as user_id/VIN', icon="ℹ️")
        pk_metadata=st.selectbox('Select your primary key to generate metadata',options=list(dataframe.columns))
        st.write(pk_metadata)
        lr_rate=st.number_input('Input the learning rate',min_value=0.01)
        st.write(lr_rate)
        latent_dims=st.number_input('Input the latent dimension',min_value=1)
        st.write(latent_dims)
        select_epoch=st.number_input('Input the epoch',min_value=1)
        st.write(select_epoch)
        select_batchsize=st.number_input('Input the batch size',min_value=1)
        st.write(select_batchsize)
        samples = st.number_input('How many data points you want to generate?',min_value=1)
        st.write(samples)
        
        if st.button('Generate and Save Synthetic Data'):
            start_time = time.time()
            with st.spinner('Loading...'):
                 time.sleep(78)
            st.write('For VAE-CVAE')
            one_hot_encoder,condition_encoder,dataframe,features,condition_features,select_input_column,select_conditional_column,encoder,decoder,latent_dims,condition_data,encoded_features = vae_cvae_synthetic_generation(dataframe,select_input_column,select_conditional_column,lr_rate,latent_dims,select_epoch,select_batchsize)
            synthetic = generate_synthetic_data(one_hot_encoder,condition_encoder,samples,dataframe,features,condition_features,select_input_column,select_conditional_column,encoder,decoder,latent_dims,condition_data,encoded_features)
            
        
            st.write(synthetic)
            st.write('Shape of synthetic data:',synthetic.shape),
            synthetic.to_csv('synthetic_data_cvae.csv', index=False)
            st.success(' VAE_CVAE Synthetic Data generated and saved successfully.')

            st.write('For VAE')
            # select_input_column_vae = st.multiselect('Select input columns for VAE',options=list(dataframe.columns),key="2")
            # st.write(select_input_column_vae)
            enc,encoded_cols,df,select_input_column_vae,encoder,decoder,latent_dim=vae_generated_synthetic_data(dataframe,select_input_column_vae,lr_rate,latent_dims,select_epoch,select_batchsize)
            synthetic_vae=generate_synthetic_data_vae(enc,encoded_cols,samples,dataframe,select_input_column_vae,encoder,decoder,latent_dims)
            st.write(synthetic_vae)
            st.write('Shape of synthetic data vae:',synthetic_vae.shape)
            synthetic_vae.to_csv('synthetic_data_vae.csv', index=False)
            st.success(' VAE Synthetic Data generated and saved successfully.')

            st.write('For CopulaGAN')
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data=dataframe)
            metadata.validate()
            synthesizer = CopulaGANSynthesizer(metadata)
            synthetic_copula=copulagan(dataframe,synthesizer,samples)
            st.write(synthetic_copula)
            st.write('Shape of synthetic data copulagan:',synthetic_copula.shape)
            synthetic_copula.to_csv('synthetic_data_copula.csv', index=False)
            st.success(' Copula Synthetic Data generated and saved successfully.')

            st.write('For FAST_ML')
            synthesizer_fastml = SingleTablePreset(metadata, name='FAST_ML')
            synthetic_fastml=fast_ml(dataframe,synthesizer_fastml,samples)
            st.write(synthetic_fastml)
            st.write('Shape of synthetic data fast_ml:',synthetic_fastml.shape)
            synthetic_fastml.to_csv('synthetic_data_fastml.csv', index=False)
            st.success(' Fast_ML Synthetic Data generated and saved successfully.')

            st.write('GaussianCopula')
            synthesizer_gaussian = GaussianCopulaSynthesizer(metadata)
            synthetic_gaussian=gaussian_copula(dataframe,synthesizer_gaussian,samples)
            st.write(synthetic_gaussian)
            st.write('Shape of synthetic data gaussian copula:',synthetic_gaussian.shape)
            synthetic_fastml.to_csv('synthetic_data_gaussian_copula.csv', index=False)
            st.success(' Gaussian Copula Synthetic Data generated and saved successfully.')

            st.write('CTGAN')
            synthesizer_ctgan = CTGANSynthesizer(metadata)
            synthetic_ctgan=ctgan(dataframe,synthesizer_ctgan,samples)
            st.write(synthetic_ctgan)
            st.write('Shape of synthetic data CTGAN:',synthetic_ctgan.shape)
            synthetic_ctgan.to_csv('synthetic_data_CTGAN.csv', index=False)
            st.success('CTGAN Synthetic Data generated and saved successfully.')

            st.write('TVAE')
            synthesizer_tvae = TVAESynthesizer(metadata)
            synthetic_tvae=tvae(dataframe,synthesizer_tvae,samples)
            st.write(synthetic_tvae)
            st.write('Shape of synthetic data TVAE:',synthetic_tvae.shape)
            synthetic_tvae.to_csv('synthetic_data_TVAE.csv', index=False)
            st.success('TVAE Synthetic Data generated and saved successfully.')



             



            end_time = time.time()
            execution_time = end_time - start_time
            st.write(f"Execution time: {execution_time:.2f} seconds")













        # if st.button('Check Quality Score'):
        #         dataframe=dataframe.drop('VIN',axis=1)
        #         synthetic=pd.read_csv('synthetic_data_cvae.csv')
        #         synthetic=synthetic.drop('VIN',axis=1)
        #         metadata = SingleTableMetadata()
        #         metadata.detect_from_dataframe(data=dataframe)
        #         metadata.validate()
        #         quality_report = evaluate_quality(
        #             dataframe,
        #             synthetic,
        #             metadata
        #         )
                # st.write(quality_report)
                # st.write(quality_report.get_score())
                # st.write(quality_report.get_details(property_name='Column Shapes'))




        if st.button('Check Quality Score'):
                start_time = time.time()
                dataframe=dataframe.drop('VIN',axis=1)
                synthetic=pd.read_csv('synthetic_data_cvae.csv')
                synthetic=synthetic.drop('VIN',axis=1)

                synthetic1=pd.read_csv('synthetic_data_vae.csv')
                synthetic1=synthetic1.drop('VIN',axis=1)

                synthetic2=pd.read_csv('synthetic_data_copula.csv')
                synthetic2=synthetic2.drop('VIN',axis=1)

                synthetic3=pd.read_csv('synthetic_data_fastml.csv')
                synthetic3=synthetic3.drop('VIN',axis=1)

                synthetic4=pd.read_csv('synthetic_data_gaussian_copula.csv')
                synthetic4=synthetic4.drop('VIN',axis=1)

                synthetic5=pd.read_csv('synthetic_data_CTGAN.csv')
                synthetic5=synthetic5.drop('VIN',axis=1)

                synthetic6=pd.read_csv('synthetic_data_TVAE.csv')
                synthetic6=synthetic6.drop('VIN',axis=1)


                metadata = {
                    "primary_key": pk_metadata,
                    "columns": {}
                    }
                for _, row in dataframe.iterrows():
                    for column_name, value in row.items():
                        if column_name != metadata["primary_key"]:
                            if column_name not in metadata["columns"]:
                                column_type = "numerical" if pd.api.types.is_numeric_dtype(dataframe[column_name]) else "categorical"
                                metadata["columns"][column_name] = {"sdtype": column_type}
                metadata_json = json.dumps(metadata, indent=4)
                # st.write(metadata_json)
                report = QualityReport()
                # q_report=report.generate(dataframe,
                #  synthetic, 
                #  metadata)
                # st.write(q_report.get_score()).
                # rg=report.generate(dataframe, synthetic, metadata)
                # st.write(rg)
                
                report.generate(dataframe, synthetic, metadata)
                a=(f"{round((report.get_score()) * 100)}%")
                # st.write(a)
                # st.write('Quality report VAE_CVAE',report.get_details(property_name='Column Shapes'))
                x1=report.get_details(property_name='Column Shapes')

                # report2 = QualityReport()
                report.generate(dataframe, synthetic1, metadata)
                b=(f"{round((report.get_score()) * 100)}%")
                # st.write(b)
                # st.write('Quality report VAE',report.get_details(property_name='Column Shapes'))
                x2=report.get_details(property_name='Column Shapes')

                report.generate(dataframe, synthetic2, metadata)
                c=(f"{round((report.get_score()) * 100)}%")
                # st.write(c)
                # st.write('Quality report CopulaGAN',report.get_details(property_name='Column Shapes'))
                x3=report.get_details(property_name='Column Shapes')

                report.generate(dataframe, synthetic3, metadata)
                d=(f"{round((report.get_score()) * 100)}%")
                # st.write(d)
                # st.write('Quality report FAST_ML',report.get_details(property_name='Column Shapes'))
                x4=report.get_details(property_name='Column Shapes')

                report.generate(dataframe, synthetic4, metadata)
                e=(f"{round((report.get_score()) * 100)}%")
                # st.write(e)
                # st.write('Quality report Gaussian Copula',report.get_details(property_name='Column Shapes'))
                x5=report.get_details(property_name='Column Shapes')

                report.generate(dataframe, synthetic5, metadata)
                f=(f"{round((report.get_score()) * 100)}%")
                # st.write(f)
                # st.write('Quality report CTGAN',report.get_details(property_name='Column Shapes'))
                x6=report.get_details(property_name='Column Shapes')

                report.generate(dataframe, synthetic6, metadata)
                g=(f"{round((report.get_score()) * 100)}%")
                # st.write(g)
                # st.write('Quality report TVAE',report.get_details(property_name='Column Shapes'))
                x7=report.get_details(property_name='Column Shapes')
                # check_winner = {'VAE-CVAE': a, 'VAE': b, 'CopulaGAN': c, 'Fast_ML': d,'Gaussian Copula':e,'CTGAN':f,'TVAE':g}
                # winner = max(check_winner, key=check_winner.get)
                # st.markdown("<h4 style='text-align: left; color: #e4b016;'>The best quality score is: {}</h4>".format(winner), unsafe_allow_html=True)



                
                scores_df = pd.DataFrame({ 'Model': ['VAE_CVAE', 'VAE', 'CopulaGAN', 'FAST_ML', 'Gaussian Copula', 'CTGAN', 'TVAE'], 'Quality Score': [a, b, c, d, e, f, g] })  
                # st.dataframe(scores_df)
                # scores_df = scores_df.sort_values('Quality Score')
                # st.line_chart(scores_df)
                # st.line_chart(scores_df.drop(['Model'], axis=1))
                scores_df_sorted = scores_df.sort_values('Quality Score',ascending= False)
                st.dataframe(scores_df_sorted)
                # st.line_chart(scores_df_sorted, x='Model', y='Quality Score')
                # scores_df_sorted['Quality Score'] = scores_df_sorted['Quality Score'].sort_index(ascending=True)
                # scores_df_sorted['Quality Score'] = scores_df_sorted['Quality Score'][::-1] 
                st.line_chart(scores_df_sorted, y='Model', x='Quality Score')

                # st.line_chart(scores_df,x='Model',y='Quality Score')


                model_names = ['VAE_CVAE', 'VAE', 'CopulaGAN', 'FAST_ML', 'Gaussian Copula', 'CTGAN', 'TVAE']
                reshaped_df = pd.DataFrame(columns=['Column', 'Metric'])
                dfs = [x1, x2, x3, x4, x5, x6, x7]
                for df, model in zip(dfs, model_names):
                    df['Quality Score of ' + model] = df['Quality Score']
                    reshaped_df = pd.concat([reshaped_df, df['Quality Score of ' + model]], axis=1)
                reshaped_df['Column'] = dfs[0]['Column']
                reshaped_df['Metric'] = dfs[0]['Metric']
                st.write(reshaped_df)
                var=reshaped_df['Column']
                reshaped_df=reshaped_df.set_index('Column')
                reshaped_df=reshaped_df.drop(columns='Metric', axis=1)
                st.line_chart(reshaped_df)

                # st.line_chart(reshaped_df.drop[]))



                check_winner = {'VAE-CVAE': a, 'VAE': b, 'CopulaGAN': c, 'Fast_ML': d,'Gaussian Copula':e,'CTGAN':f,'TVAE':g}
                winner = max(check_winner, key=check_winner.get)
                st.markdown("<h4 style='text-align: left; color: #e4b016;'>The best quality score is: {}</h4>".format(winner), unsafe_allow_html=True)
                end_time = time.time()
                execution_time = end_time - start_time
                st.write(f"Execution time: {execution_time:.2f} seconds")
                





                


                







       










        
        

 
