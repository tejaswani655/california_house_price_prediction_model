import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle
import time

# Title

col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

st.title('California Housing Price Prediction')

st.image('https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.linkedin.com%2Fpulse%2Fexploring-house-price-prediction-data-analysis-journey-raymond-ng&psig=AOvVaw2_oEvsyeUqMsl95ZkHMj75&ust=1756014381647000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCMCCjL-doI8DFQAAAAAdAAAAABAE')



st.header('Model of housing prices to predict median house values in California ',divider=True)

st.subheader('''User Must Enter Given values to predict Price:
['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')


st.sidebar.title('Select House Features 🏠')

st.sidebar.image('https://miro.medium.com/v2/resize:fit:1024/0*YMZOAO8QE4bZ4_Rk.jpg')


# read_data
temp_df = pd.read_csv('california.csv')

random.seed(52)

st.markdown('Designed by:**Tejaswani Raj**')

all_values = []

for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])

    var =st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)

ss = StandardScaler()
ss.fit(temp_df[col])

final_value = ss.transform([all_values])

with open('house_price_pred_ridge_model.pkl','rb') as f:
    chatgpt = pickle.load(f)

price = chatgpt.predict(final_value)[0]


import time


progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Price')

if price>0:

    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i + 1)

    body = f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'
    placeholder.empty()
    # st.subheader(body)

    st.success(body)
else:
    body = 'Invalid House features Values'
    st.warning(body)
