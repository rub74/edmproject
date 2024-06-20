import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import altair as alt
import matplotlib.pyplot as plt


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
