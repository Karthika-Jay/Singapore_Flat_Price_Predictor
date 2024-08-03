# Packages

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
import streamlit as st
import pickle
import numpy as np
from datetime import date
import sklearn
from streamlit_option_menu import option_menu
import base64
from sklearn.ensemble import RandomForestRegressor
from joblib import dump


def town_mapping(town_map):
    if town_map == 'ANG MO KIO':
        town_1 = int(0)
    elif town_map == 'BEDOK':
        town_1 = int(1)
    elif town_map == 'BISHAN':
        town_1= int(2)
    elif town_map == 'BUKIT BATOK':
        town_1= int(3)
    elif town_map == 'BUKIT MERAH':
        town_1= int(4)
    elif town_map == 'BUKIT PANJANG':
        town_1= int(5)

    elif town_map == 'BUKIT TIMAH':
        town_1= int(6)
    elif town_map == 'CENTRAL AREA':
        town_1= int(7)
    elif town_map == 'CHOA CHU KANG':
        town_1= int(8)
    elif town_map == 'CLEMENTI':
        town_1= int(9)
    elif town_map == 'GEYLANG':
        town_1= int(10)
    
    elif town_map == 'HOUGANG':
        town_1 = int(11)
    elif town_map == 'JURONG EAST':
        town_1= int(12)
    elif town_map == 'JURONG WEST':
        town_1= int(13)
    elif town_map == 'KALLANG/WHAMPOA':
        town_1= int(14)
    elif town_map == 'MARINE PARADE':
        town_1= int(15)

    elif town == 'PASIR RIS':
        town_1= int(16)
    elif town == 'PUNGGOL':
        town_1= int(17)
    elif town == 'QUEENSTOWN':
        town_1= int(18)
    elif town == 'SEMBAWANG':
        town_1= int(19)
    elif town == 'SENGKANG':
        town_1= int(20)

    elif town == 'SERANGOON':
        town_1= int(21)
    elif town == 'TAMPINES':
        town_1= int(22)
    elif town == 'TOA PAYOH':
        town_1= int(23)
    elif town == 'WOODLANDS':
        town_1= int(24)        
    elif town == 'YISHUN':
        town_1= int(25)      

    return town_1

def flat_type_mapping(flt_type):

    if flt_type == '3 ROOM':
        flat_type_1= int(2)
    elif flt_type == '4 ROOM':
        flat_type_1= int(3)
    elif flt_type == '5 ROOM':
        flat_type_1= int(4)
    elif flt_type == '2 ROOM':
        flat_type_1= int(1)
    elif flt_type == 'EXECUTIVE':
        flat_type_1= int(5)
    elif flt_type == '1 ROOM':
        flat_type_1= int(0)
    elif flt_type == 'MULTI-GENERATION':
        flat_type_1= int(6)

    return flat_type_1

def flat_model_mapping(fl_m):

    if fl_m == 'Improved':
        flat_model_1= int(5)
    elif fl_m == 'New Generation':
        flat_model_1= int(12)
        
    elif fl_m == 'Model A':
        flat_model_1= int(8)
    elif fl_m == 'Standard':
        flat_model_1= int(17)
    elif fl_m == 'Simplified':
        flat_model_1= int(16)
    elif fl_m == 'Premium Apartment':
        flat_model_1= int(13)
    elif fl_m == 'Maisonette':
        flat_model_1= int(7)

    elif fl_m == 'Apartment':
        flat_model_1= int(3)
    elif fl_m == 'Model A2':
        flat_model_1= int(10)
    elif fl_m == 'Type S1':
        flat_model_1= int(19)
    elif fl_m == 'Type S2':
        flat_model_1= int(20)
    elif fl_m == 'Adjoined flat':
        flat_model_1= int(2)

    elif fl_m == 'Terrace':
        flat_model_1= int(18)
    elif fl_m == 'DBSS':
        flat_model_1= int(4)
    elif fl_m == 'Model A-Maisonette':
        flat_model_1= int(9)
    elif fl_m == 'Premium Maisonette':
        flat_model_1= int(15)
    elif fl_m == 'Multi Generation':
        flat_model_1= int(11)

    elif fl_m == 'Premium Apartment Loft':
        flat_model_1= int(14)
    elif fl_m == 'Improved-Maisonette':
        flat_model_1= int(6)
    elif fl_m == '2-room':
        flat_model_1= int(0)
    elif fl_m == '3Gen':
        flat_model_1= int(1)

    return flat_model_1


def predict_price(year,town,flat_type,flr_area_sqm,flat_model,stry_start,stry_end,re_les_year,
              re_les_month,les_coms_dt):
    
    

    year_1 = int(year)
    town_2 = town_mapping(town)
    flt_ty_2 = flat_type_mapping(flat_type)
    flr_ar_sqm_1 = int(flr_area_sqm)
    flt_model_2 = flat_model_mapping(flat_model)

    # Validate inputs
    if flr_ar_sqm_1 <= 0 or flr_ar_sqm_1 > 280:
        raise ValueError("Floor area must be between 1 and 280 sqm.")
    if stry_start <= 0 or stry_end <= 0:
        raise ValueError("Storey start and end must be greater than 0.")
    
    str_str = np.log(stry_start) if stry_start > 0 else 0
    str_end = np.log(stry_end) if stry_end > 0 else 0

    rem_les_year = int(re_les_year)
    rem_les_month = int(re_les_month)
    lese_coms_dt = int(les_coms_dt)


    with open('C:/Users/lenovo/Desktop/try/singapore_flat_resale_analysis/Resale_prediction_Model_1.pkl', 'rb') as f:
        regg_model = pickle.load(f)


    user_data = np.array([[year_1,town_2,flt_ty_2,flr_ar_sqm_1,
                           flt_model_2,str_str,str_end,rem_les_year,rem_les_month,
                           lese_coms_dt]])
    y_pred_1 = regg_model.predict(user_data)
    price= np.exp(y_pred_1[0])

    return round(price)



st.set_page_config(layout="wide")


st.markdown("<div style='background-color:rgb(0, 0, 0);padding:10px;text-align:center;'><h1 style='color:white;'>SINGAPORE RESALE FLAT PRICES PREDICTION</h1></div>", unsafe_allow_html=True)

st.write("")

tabs = st.tabs(["**🏠 Home**", "**🔍 Price Prediction**"])

with tabs[0]:
    # Load the image and encode it in base64
    with open("1.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    # CSS to set the background image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_string});
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    
# Center the title using HTML and CSS
    st.markdown(
    """
    <style>
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0; }
        100% { opacity: 1; }
    }
    .blink {
        animation: blink 1s infinite;
        font-size: 3em; 
        color: rainbow(10); 
        text-align: center; 
        display: block;
    }
    .container {
        text-align: center; 
    }
    </style>
    <div class='container'>
        <div class='blink'>🏠🆆🅴🅻🅲🅾🅼🅴🏘</div>
    </div>
    """,
    unsafe_allow_html=True
)

    
    col1, col2= st.columns([1, 1])
    with col1:
        st.image("6.gif")

        st.markdown('''<div style="background-color:white;padding:6px;text-align:left;"><p style="color:black;font-size:20px;">

**OVERVIEW**:
                    
The Singapore Resale Flat Prices Predicting project aims to leverage machine learning techniques to predict the resale prices of flats in Singapore. This project is designed to assist both potential buyers and sellers in making informed decisions by providing an estimated resale value based on historical data and various influencing factors.

**KEY COMPONENTS**:

🔹Data Collection and Preprocessing

**Data Source**: The dataset comprises historical resale flat transactions obtained from the Singapore Housing and Development Board (HDB) spanning from 1990 to the present. 
                    
**Preprocessing Steps**: Cleaning and structuring the data to make it suitable for machine learning. This includes handling missing values, encoding categorical variables, and normalizing numerical features.
                    
🔹Feature Engineering: 

Identifying and extracting relevant features from the dataset such as:
                    
    Town: The area or district where the flat is located.
                    
    Flat Type: The type of flat (e.g., 3-room, 4-room, etc.).
                    
    Storey Range: The floor level range of the flat.
                    
    Floor Area: The size of the flat in square meters.
                    
    Flat Model: The model or design of the flat.
                    
    Lease Commence Date: The year the lease of the flat commenced.
                    
Creating additional features to enhance the model's predictive accuracy. 
                                   
🔹Model Evaluation

Assessing the model's performance using regression metrics including:
                    
    Mean Absolute Error (MAE)
                        
    Mean Squared Error (MSE)
                        
    Root Mean Squared Error (RMSE)
                        
    R-squared (R²) Score
                    
🔹Web Application Development

Creating a user-friendly web application using Streamlit. The app will allow users to input various details about a flat (e.g., town, flat type, storey range) and receive a predicted resale price.

The application will integrate the trained machine learning model to provide real-time predictions.
        
🔹Deployment

Deploying the web application on a platform like Render to make it accessible to users online.        
                    
🔹Testing and Validation

Thoroughly testing the deployed application to ensure it functions correctly and provides accurate predictions.
                    
Gathering feedback from users to further refine and improve the model and application.
                
🔹 Results and Benefits
                    
For Buyers: The application helps potential buyers estimate resale prices, enabling them to make informed purchasing decisions.

For Sellers: Sellers can use the application to get an idea of their flat's potential market value, aiding in pricing their property competitively.

General Impact: Demonstrates the practical application of machine learning in real estate, showcasing how data-driven approaches can streamline decision-making processes.
        

                
By combining historical data analysis with modern machine learning techniques, the Singapore Resale Flat Prices Predicting project offers a robust tool for navigating the competitive real estate market in Singapore.                </p></div>''', unsafe_allow_html=True)
    # Create a centered column
    
    with col2:
        st.write(" ")
        st.write(" ")
        st.write(" ")

        st.image("3.gif")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.image("10.gif")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")

        st.image("4.gif")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.image("9.gif")


        
with tabs[1]:
    
# Center the title using HTML and CSS
    
    st.markdown("<div style='background-color:blue;padding:5px;text-align:center;'><h1 style='color:white;'>💰Predicting Singapore Flat Price💸</h1></div>", unsafe_allow_html=True)
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.image("5.gif")

    col1,col2= st.columns(2)
    with col1:

        year= st.selectbox(":rainbow[**Select the Year**]",["2015", "2016", "2017", "2018", "2019", "2020", "2021",
                           "2022", "2023", "2024"])
        
        town= st.selectbox(":rainbow[**Select the Town**]", ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
                                            'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
                                            'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
                                            'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
                                            'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
                                            'TOA PAYOH', 'WOODLANDS', 'YISHUN'])
        
        flat_type= st.selectbox(":rainbow[**Select the Flat Type**]", ['3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE', '1 ROOM',
                                                        'MULTI-GENERATION'])
        
        flr_area_sqm= st.number_input(":rainbow[**Enter the Value of Floor Area sqm Min: 31 / Max: 280**]")

        flat_model= st.selectbox(":rainbow[**Select the Flat Model**]", ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified',
                                                        'Premium Apartment', 'Maisonette', 'Apartment', 'Model A2',
                                                        'Type S1', 'Type S2', 'Adjoined flat', 'Terrace', 'DBSS',
                                                        'Model A-Maisonette', 'Premium Maisonette', 'Multi Generation',
                                                        'Premium Apartment Loft', 'Improved-Maisonette', '2-room', '3Gen'])
        
    with col2:

        stry_start= st.number_input(":rainbow[**Enter the Value of Storey Start Min: 0.011858 / Max: 3.677021**]")

        stry_end= st.number_input(":rainbow[**Enter the Value of Storey End Min: 1.098612 / Max: 3.524627**]")

        re_les_year= st.number_input(":rainbow[**Enter the Value of Remaining Lease Year Min: 42 / Max: 97**]")

        re_les_month= st.number_input(":rainbow[**Enter the Value of Remaining Lease Month Min: 0 / Max: 11**]")
        
        les_coms_dt= st.selectbox(":rainbow[**Select the Lease_Commence_Date**]", [str(i) for i in range(1966,2023)])

    button= st.button(":rainbow[**Predict the Price**]", use_container_width= True)

    if button:
        pre_price= predict_price(year, town, flat_type, flr_area_sqm, flat_model,
                        stry_start, stry_end, re_les_year, re_les_month, les_coms_dt)


        # Display the predicted price with padding and blinking effect
        st.markdown(
            f"""
            <div style='padding: 20px; font-size: 35px; color: blue; text-align: center;'>
                <span style='animation: blink 1s steps(5, start) infinite;'> 
                    <strong>The Predicted Price is: 💲{pre_price}</strong>
                </span>
            </div>
            <style>
            @keyframes blink {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0; }}
                100% {{ opacity: 1; }}
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        col1, col2, col3= st.columns([3, 3, 1])
        with col2:
            st.image("7.gif")
            st.balloons()