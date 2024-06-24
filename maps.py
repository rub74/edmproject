import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm
from stqdm import stqdm


data = pd.read_csv('listings_redux.csv')


st.header("AirBnB distribution in Madrid")

midpoint = (np.average(data["latitude"]), np.average(data["longitude"]))
st.write(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state={
        "latitude": midpoint[0],
        "longitude": midpoint[1],
        "zoom": 11,
        "pitch": 50,
    },
    layers=[
        pdk.Layer(
            "HexagonLayer",
            data=data[['latitude', 'longitude']],
            get_position=["longitude", "latitude"],
            auto_highlight=True,
            radius=100,
            extruded=True,
            pickable=True,
            elevation_scale=4,
            elevation_range=[0, 1000],
        ),
    ],
))

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt

# Leer los datos del archivo CSV
data = pd.read_csv('listings_redux.csv')

# Guardar cada columna en variables separadas
latitude = data['latitude']
longitude = data['longitude']
neighbourhood_group = data['neighbourhood_group']
neighbourhood = data['neighbourhood']
price = data['price']
minimum_nights = data['minimum_nights']
availability_365 = data['availability_365']

# Calcular el punto medio
midpoint = (np.average(latitude), np.average(longitude))

# Configurar y mostrar el mapa
st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state={
        "latitude": midpoint[0],
        "longitude": midpoint[1],
        "zoom": 11,
        "pitch": 50,
    },
    layers=[
        pdk.Layer(
            "HexagonLayer",
            data=data,
            get_position=["longitude", "latitude"],
            auto_highlight=True,
            radius=200,
            extruded=True,
            pickable=True,
            elevation_scale=4,
            elevation_range=[0, 1000],
        ),
    ],
    tooltip={
        "html": "<b>Neighbourhood Group:</b> {neighbourhood_group}<br/>"
                "<b>Neighbourhood:</b> {neighbourhood}<br/>"
                "<b>Price:</b> ${price}<br/>"
                "<b>Minimum Nights:</b> {minimum_nights}<br/>"
                "<b>Availability:</b> {availability_365} days",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white",
            "fontSize": "14px",
            "padding": "10px"
        }
    }
))


st.header("Airbnb locations by minimum price")

price = st.slider("Price per night", min(data['price']), max(data['price']))
st.map(data.query("price >= @price")[["latitude", "longitude"]].dropna(how="any"))

st.header("Most expensive neighbourhoods by cummulative price")

cumulative_price_df = data[['neighbourhood', 'price']].groupby('neighbourhood')['price'].sum().reset_index()
n = st.slider('Select the number of top neighbourhoods by price:', 1, len(cumulative_price_df), 5)

top_n_df = cumulative_price_df.sort_values(by='price', ascending=False).head(n)

# Bar chart with plt
#fig, ax = plt.subplots()
#ax.bar(top_n_df['neighbourhood'], top_n_df['price'], color='skyblue')
#ax.set_xlabel('Neighbourhood')
#ax.set_ylabel('Cumulative Price')
#ax.set_title('Top Neighbourhoods by Cumulative Price')

#st.pyplot(fig)

fig2 = alt.Chart(top_n_df).mark_bar().encode(
    x=alt.X('neighbourhood', sort=None),
    y='price'
)
st.write(fig2)

st.header("Maximum and minimum price by neighbourhood")

# Neighbourhood selection
neighbourhoods = data['neighbourhood'].unique()
selected_neighbourhood = st.selectbox('Select a neighbourhood:', neighbourhoods)

# Filter data for the selected neighbourhood
filtered_df = data[data['neighbourhood'] == selected_neighbourhood]

# Calculate max and min price
max_price = filtered_df['price'].max()
min_price = filtered_df['price'].min()

# Display metrics
col1, col2 = st.columns(2)
col1.metric(label=f"Maximum Price in {selected_neighbourhood}", value=f"{max_price}€")
col2.metric(label=f"Minimum Price in {selected_neighbourhood}", value=f"{min_price}€")


st.header('Price Comparison Between Neighbourhoods')

# Neighbourhood selection
neighbourhoods = data['neighbourhood'].unique()
neighbourhood1 = st.selectbox('Select the first neighbourhood:', neighbourhoods, index=0)
neighbourhood2 = st.selectbox('Select the second neighbourhood:', neighbourhoods, index=1)

# Filter data for the selected neighbourhoods
filtered_df1 = data[data['neighbourhood'] == neighbourhood1]
filtered_df2 = data[data['neighbourhood'] == neighbourhood2]

# Calculate max prices for the selected neighbourhoods
max_price1 = filtered_df1['price'].max()
max_price2 = filtered_df2['price'].max()
min_price1 = filtered_df1['price'].min()
min_price2 = filtered_df2['price'].min()
# Calculate the price delta
price_delta_max = max_price2 - max_price1
price_delta_min = min_price2 - min_price1
# Display metrics
col1, col2 = st.columns(2)
col1.metric(label=f"Maximum Price in {neighbourhood1}", value=f"${max_price1}")
col1.metric(label=f"Minimum Price in {neighbourhood1}", value=f"${min_price1}")

col2.metric(label=f"Maximum Price in {neighbourhood2}", value=f"${max_price2}", delta=float(price_delta_max), delta_color="inverse")
col2.metric(label=f"Minimum Price in {neighbourhood2}", value=f"${min_price2}", delta=float(price_delta_min), delta_color="inverse")



# Load XGBoost classifier
# To deserialize estimator later

with open('our_estimator.pkl', 'rb') as fid:
    pp = pickle.load(fid)


st.header('Price prediction for upcoming dates')
st.subheader('According to neighbourhoods')

df = pd.read_csv('cal_listings_merged.csv')
features = ['year', 'month', 'day', 'dayofweek', 'is_weekend', 'quarter', 'is_holiday', 'days_since_start', 'last_year_price', 'mean_airbnb_price']
target = 'log_price'


# Date input widget
start_date = st.date_input("Select a start date", datetime.today())
end_date = st.date_input("Select an end date", datetime.today() + timedelta(days=7))

# Neighborhood selection
neighborhoods = df['neighbourhood'].unique()
selected_neighborhood = st.selectbox("Select a Neighborhood", neighborhoods)

# Filter data based on selected neighborhood
filtered_df = df[df['neighbourhood'] == selected_neighborhood]


def create_features(start_date, end_date, dfr):
    future_dates = pd.date_range(start_date, end_date)

    since_date = datetime.strptime(dfr['date'].min(), '%Y-%m-%d')

    def days_difference(date1, date2):
        d1 = datetime.combine(date1, datetime.min.time())
        d2 = datetime.combine(date2, datetime.min.time())
        delta = date1 - date2
        return abs(delta.days)

    

# Convierte future_dates a un DataFrame y extrae año, mes, día, etc.
    future_dates_df = pd.DataFrame({'date': future_dates})
    future_dates_df['year'] = future_dates_df['date'].dt.year
    future_dates_df['month'] = future_dates_df['date'].dt.month
    future_dates_df['day'] = future_dates_df['date'].dt.day
    future_dates_df['dayofweek'] = future_dates_df['date'].dt.dayofweek
    future_dates_df['is_weekend'] = future_dates_df['dayofweek'].isin([5, 6]).astype(int)
    future_dates_df['quarter'] = future_dates_df['date'].dt.quarter
    future_dates_df['is_holiday'] = ((future_dates_df['month'] == 12) & (future_dates_df['day'] >= 20)) | ((future_dates_df['month'] == 1) & (future_dates_df['day'] <= 5))
    future_dates_df['days_since_start'] = (future_dates_df['date'] - since_date).astype(int)
    
    return future_dates_df


# Predict button
future_dates_df = create_features(start_date, end_date, filtered_df)
airbnb_mean_prices = filtered_df.groupby('listing_id')['price'].mean()

    # Crea un DataFrame con las fechas futuras replicando cada Airbnb para cada fecha
airbnb_ids = filtered_df['listing_id'].unique()
future_data = pd.DataFrame()

if st.button("Predict"):
    for airbnb_id in stqdm(airbnb_ids, desc="Predicting prices", total=len(airbnb_ids)):
            airbnb_data = future_dates_df.copy()
            airbnb_data['airbnb_id'] = airbnb_id
            
            # Prepara los datos del año pasado
            past_data = filtered_df[filtered_df['listing_id'] == airbnb_id].copy()
            past_data['month_day'] = pd.to_datetime(past_data['date']).dt.strftime('%m-%d')
            #past_data['month_day'] = past_data['month_day'].values.astype("float64")
            
            # Merge para agregar la característica del precio del año pasado en el mismo día y mes
            airbnb_data['month_day'] = pd.to_datetime(airbnb_data['date']).dt.strftime('%m-%d')
            #airbnb_data['month_day'] = airbnb_data['month_day'].values.astype("float64")
            merged_data = pd.merge(airbnb_data, past_data[['month_day', 'price']], on='month_day', how='left')
            airbnb_data['last_year_price'] = merged_data['price'].fillna(airbnb_mean_prices[airbnb_id])
            
            # Añade la media del precio del Airbnb como una característica
            airbnb_data['mean_airbnb_price'] = airbnb_mean_prices[airbnb_id]
            
            # Asegúrate de que todas las características necesarias estén presentes
            for feature in features:
                if feature not in airbnb_data.columns:
                    airbnb_data[feature] = 0  # o cualquier otro valor por defecto apropiado
            # Hacer predicciones para este Airbnb y las fechas futuras
            predicted_log_prices = pp.predict(airbnb_data[features])
            
            # Convertir el logaritmo del precio predicho a la escala original
            predicted_prices = np.expm1(predicted_log_prices)
            
            # Agregar las predicciones al DataFrame de fechas futuras
            airbnb_data['predicted_price'] = predicted_prices
            
            # Concatenar los resultados al DataFrame principal
            future_data = pd.concat([future_data, airbnb_data], ignore_index=True)

        # Mostrar las predicciones para todas las combinaciones de Airbnb y fechas futuras

    filtered_data = future_data[['airbnb_id', 'date', 'predicted_price']]
    st.write(filtered_data)


