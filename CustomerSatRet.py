import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Loading the trained model
model_ret = tf.keras.models.load_model('model_ret.h5')
model_sat = tf.keras.models.load_model('model_sat.h5')

### Load the encoders and scaler
with open('label_encoder_six_col.pk1','rb') as file:
    label_encoder_six_col=pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('scaler_ret.pkl','rb') as file:
    scaler_ret=pickle.load(file)

with open('scaler_sat.pkl','rb') as file:
    scaler_sat=pickle.load(file)

## Streamlit app
st.title('Customer Satisfaction & Retention Prediction')

## User inputs
customer_claims = st.number_input('Customer Claims per Month', format="%i")
incident_count = st.number_input('Incident Count', format="%i")
inc_rel_in_time = st.selectbox('Incident Resolution in Time (Yes or No):', label_encoder_six_col.classes_)
in_time_payment = st.selectbox('On time payment by Customer (Yes or No):', label_encoder_six_col.classes_)
inv_on_time = st.selectbox('Invoice sent on time (Yes or No):', label_encoder_six_col.classes_)
is_eu_vat_issue = st.selectbox('If EU customer, any VAT issues (Yes or No):', label_encoder_six_col.classes_)
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
is_active = st.selectbox('Is active Customer (Yes or No):', label_encoder_six_col.classes_)

## Prepare the input data
input_data = pd.DataFrame ({
    'CUSTOMER_CLAIMS': [customer_claims],
    'INCIDENT_COUNT': [incident_count],
    'INCIDENT_RESOLVED_IN_TIME': [inc_rel_in_time],
    'IN_TIME_PAYMENT': [in_time_payment],
    'INVOICE_ON_TIME': [inv_on_time],
    'EU_VAT_ISSUE': [is_eu_vat_issue],
    'GEOGRAPHY': [geography],
    'ACTIVE_FLAG': [is_active]
})

# One-got Encoder for 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]])
geo_encoded_df=pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['GEOGRAPHY']))

# Combine onehot encoded data to input data
#input_data_ret=pd.concat([input_data.drop('CUSTOMER_SATISFACTION',axis=1),
#                         input_data.reset_index(drop=True),geo_encoded_df], axis=1)
input_data=pd.concat([input_data.drop('GEOGRAPHY', axis=1),geo_encoded_df], axis=1)

# Label encoding for the six columns
label_encoder_six_col=LabelEncoder()
cols_to_encode=['INCIDENT_RESOLVED_IN_TIME', 'IN_TIME_PAYMENT', 'INVOICE_ON_TIME', 'EU_VAT_ISSUE', 
     'ACTIVE_FLAG']
for col in cols_to_encode:
    input_data[col]=label_encoder_six_col.fit_transform(input_data[col])

## Scaling the input data
input_scaled_ret=scaler_ret.transform(input_data)
input_scaled_sat=scaler_sat.transform(input_data)

## Predict Exited(churn) variable
prediction_ret=model_ret.predict(input_scaled_ret)
prediction_prob_ret = prediction_ret[0][0]

prediction_sat=model_sat.predict(input_scaled_sat)
prediction_prob_sat = prediction_sat[0][0]

# display Retention probablity
st.write(f'Retention probability: {prediction_prob_ret:.2f}')
st.write(f'Predicted Customer Satisfaction: {prediction_prob_sat:.2f}')

def colored_star_rating(rating, max_stars=5):
    """
    Renders a colored star rating using st.markdown and HTML/CSS.
    Colors change based on the rating value.
    """
    rating = max(0, min(prediction_prob_sat, max_stars)) # Ensure rating is within valid range

    # Define colors for different rating ranges
    if rating >= 4.0:
        color = "green"
    elif rating >= 2.5:
        color = "gold" # Yellow is 'gold' in CSS for better visibility
    else:
        color = "red"

    # Generate the star icons using HTML entities
    full_stars = int(rating)
    empty_stars = max_stars - full_stars
    
    stars_html = f"""
    <div style="color: {color}; font-size: 24px;">
        {'★' * full_stars}
        {'☆' * empty_stars}
    </div>
    """
    
    # Display the stars and the value using st.markdown with unsafe_allow_html=True
    st.markdown(stars_html, unsafe_allow_html=True)
    st.write(f"**Value:** {rating:.1f} out of {max_stars}")

# --- Streamlit App ---
st.title("Predicted Customer Satisfaction")
# Display the stars and the value using st.markdown with unsafe_allow_html=True
st.write(colored_star_rating(prediction_prob_sat,max_stars=5))
st.markdown("---")

# Display the result
if 0.01 < prediction_prob_ret < 0.5:
    st.write('The customer will not exit.')
elif 0.51 < prediction_prob_ret < 0.89:
    st.write('The customer is likely to exit.')
elif prediction_prob_ret > 0.95:
    st.write('The customer will most probably exit.')