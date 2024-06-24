import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import altair as alt
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor  # o el modelo que estés usando


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

st.info("This 3D map visualizes the density of AirBnB listings across Madrid. Higher hexagons indicate areas with more listings.")

st.header("Airbnb locations by minimum price")

price = st.slider("Price per night", int(min(data['price'])), int(max(data['price'])))
st.map(data.query("price >= @price")[["latitude", "longitude"]].dropna(how="any"))

st.info("This map shows the locations of AirBnB listings filtered by a minimum price. Use the slider to adjust the minimum price and see how it affects the distribution of listings.")

st.header("Median price by neighborhood")

# Calculate the median price by neighborhood
price_stats_df = data.groupby('neighbourhood')['price'].median().reset_index()
price_stats_df = price_stats_df.sort_values(by='price', ascending=False)

# Slider to select the number of neighborhoods to show
n = st.slider('Select the number of neighborhoods:', 5, min(20, len(price_stats_df)), 10)

# Select the top n neighborhoods
top_n_df = price_stats_df.head(n)

# Create a simple horizontal bar chart
chart = alt.Chart(top_n_df).mark_bar().encode(
    y=alt.Y('neighbourhood:N', sort='-x', title='Neighborhood'),
    x=alt.X('price:Q', title='Median price per night (€)'),
    color=alt.Color('price:Q', scale=alt.Scale(scheme='blues'), legend=None)
).properties(
    title='Median Price by Neighborhood',
    width=600,
    height=max(300, 30 * n)  # Adjust height based on number of neighborhoods
)

# Add text labels for prices
text = chart.mark_text(
    align='left',
    baseline='middle',
    dx=3  # Nudge text to right so it doesn't appear on top of the bar
).encode(
    text=alt.Text('price:Q', format='.2f')
)

# Combine chart and text
final_chart = (chart + text).configure_axis(
    labelFontSize=12,
    titleFontSize=14
)

# Display the chart
st.altair_chart(final_chart, use_container_width=True)
st.header("Price Analysis by Neighborhood")

# Neighborhood selection
neighbourhoods = sorted(data['neighbourhood'].unique())
selected_neighbourhood = st.selectbox('Select a neighborhood:', neighbourhoods)

# Filter data for the selected neighborhood
filtered_df = data[data['neighbourhood'] == selected_neighbourhood]

# Calculate statistics
max_price = int(filtered_df['price'].max())
min_price = int(filtered_df['price'].min())
median_price = int(filtered_df['price'].median())
avg_price = int(filtered_df['price'].mean())
count = len(filtered_df)

# Display metrics
col1, col2, col3 = st.columns(3)
col1.metric(label="Maximum Price", value=f"{max_price}€")
col1.metric(label="Minimum Price", value=f"{min_price}€")
col2.metric(label="Median Price", value=f"{median_price}€")
col2.metric(label="Average Price", value=f"{avg_price}€")
col3.metric(label="Number of Properties", value=count)

# Price distribution graph
st.subheader(f"Price Distribution in {selected_neighbourhood}")
chart = alt.Chart(filtered_df).mark_bar().encode(
    alt.X('price', bin=alt.Bin(maxbins=30), title='Price (€)'),
    y='count()',
    tooltip=['count()', 'price']
).properties(
    width=600,
    height=300
)
st.altair_chart(chart, use_container_width=True)

st.info("This section provides a detailed price analysis for a selected neighborhood. It shows key statistics (max, min, median, average prices) and a histogram of price distribution, giving you a comprehensive view of pricing in that area.")

st.header('Price Comparison Between Neighborhoods')

# Neighborhood selection for comparison
col1, col2 = st.columns(2)
with col1:
    neighbourhood1 = st.selectbox('Select the first neighborhood:', neighbourhoods, index=0)
with col2:
    neighbourhood2 = st.selectbox('Select the second neighborhood:', neighbourhoods, index=1)

# Filter data for selected neighborhoods
filtered_df1 = data[data['neighbourhood'] == neighbourhood1]
filtered_df2 = data[data['neighbourhood'] == neighbourhood2]

# Calculate statistics for both neighborhoods
stats1 = filtered_df1['price'].agg(['max', 'min', 'median', 'mean']).apply(int)
stats2 = filtered_df2['price'].agg(['max', 'min', 'median', 'mean']).apply(int)

# Display comparative metrics
col1, col2 = st.columns(2)
for stat in ['max', 'min', 'median', 'mean']:
    col1.metric(label=f"{stat.capitalize()} Price in {neighbourhood1}", value=f"{stats1[stat]}€")
    delta = stats2[stat] - stats1[stat]
    col2.metric(label=f"{stat.capitalize()} Price in {neighbourhood2}", value=f"{stats2[stat]}€", 
                delta=f"{delta}€", delta_color="inverse")

# Comparison graph
comparison_data = pd.DataFrame({
    'neighbourhood': [neighbourhood1, neighbourhood2],
    'median_price': [stats1['median'], stats2['median']]
})

chart = alt.Chart(comparison_data).mark_bar().encode(
    x='neighbourhood',
    y='median_price',
    color='neighbourhood',
    tooltip=['neighbourhood', 'median_price']
).properties(
    width=400,
    height=300,
    title='Comparison of Median Prices'
)
st.altair_chart(chart, use_container_width=True)

st.info("This comparison tool allows you to select two neighborhoods and compare their price statistics side by side. The bar chart visually represents the median prices, while the metrics show the difference in various price points between the two areas.")
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm
import joblib
import os

# Load XGBoost classifier
with open('our_estimator.pkl', 'rb') as fid:
    pp = joblib.load(fid)

st.header('Price prediction for upcoming dates')
st.subheader('According to neighbourhoods')

# Load neighborhoods
neighborhoods = pd.read_csv('neighbourhoods.csv')['neighbourhood'].tolist()

# Date input widget
start_date = st.date_input("Select a start date", datetime.today())
end_date = st.date_input("Select an end date", datetime.today() + timedelta(days=7))

# Neighborhood selection
selected_neighborhood = st.selectbox("Select a Neighborhood", neighborhoods)

# Function to load data for a specific neighborhood
def load_neighborhood_data(neighborhood):
    folder = 'cal_listings_merged'
    neighborhood_files = [f for f in os.listdir(folder) if f.startswith(neighborhood.replace(' ', '_'))]
    
    if not neighborhood_files:
        st.error(f"No data found for {neighborhood}")
        return None
    
    dfs = []
    for file in neighborhood_files:
        df = pd.read_csv(os.path.join(folder, file))
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

# Load data for selected neighborhood
filtered_df = load_neighborhood_data(selected_neighborhood)

if filtered_df is not None:
    features = ['year', 'month', 'day', 'dayofweek', 'is_weekend', 'quarter', 'is_holiday', 'days_since_start', 'last_year_price', 'mean_airbnb_price']
    target = 'log_price'

    def create_features(start_date, end_date, dfr):
        future_dates = pd.date_range(start_date, end_date)
        since_date = pd.to_datetime(dfr['date'].min())

        future_dates_df = pd.DataFrame({'date': future_dates})
        future_dates_df['year'] = future_dates_df['date'].dt.year
        future_dates_df['month'] = future_dates_df['date'].dt.month
        future_dates_df['day'] = future_dates_df['date'].dt.day
        future_dates_df['dayofweek'] = future_dates_df['date'].dt.dayofweek
        future_dates_df['is_weekend'] = future_dates_df['dayofweek'].isin([5, 6]).astype(int)
        future_dates_df['quarter'] = future_dates_df['date'].dt.quarter
        future_dates_df['is_holiday'] = ((future_dates_df['month'] == 12) & (future_dates_df['day'] >= 20)) | ((future_dates_df['month'] == 1) & (future_dates_df['day'] <= 5))
        future_dates_df['days_since_start'] = (future_dates_df['date'] - since_date).dt.days
        
        return future_dates_df

    # Predict button
    future_dates_df = create_features(start_date, end_date, filtered_df)
    airbnb_mean_prices = filtered_df.groupby('listing_id')['price'].mean()

    airbnb_ids = filtered_df['listing_id'].unique()[:50]  # Predict only for the first 50 Airbnbs
    future_data = pd.DataFrame()

    if st.button("Predict"):
        for airbnb_id in tqdm(airbnb_ids, desc="Predicting prices", total=len(airbnb_ids)):
            airbnb_data = future_dates_df.copy()
            airbnb_data['airbnb_id'] = airbnb_id
            
            past_data = filtered_df[filtered_df['listing_id'] == airbnb_id].copy()
            past_data['month_day'] = pd.to_datetime(past_data['date']).dt.strftime('%m-%d')
            
            airbnb_data['month_day'] = pd.to_datetime(airbnb_data['date']).dt.strftime('%m-%d')
            merged_data = pd.merge(airbnb_data, past_data[['month_day', 'price']], on='month_day', how='left')
            airbnb_data['last_year_price'] = merged_data['price'].fillna(airbnb_mean_prices[airbnb_id])
            
            airbnb_data['mean_airbnb_price'] = airbnb_mean_prices[airbnb_id]
            
            for feature in features:
                if feature not in airbnb_data.columns:
                    airbnb_data[feature] = 0
            
            predicted_log_prices = pp.predict(airbnb_data[features])
            predicted_prices = np.expm1(predicted_log_prices)
            
            airbnb_data['predicted_price'] = predicted_prices
            
            future_data = pd.concat([future_data, airbnb_data], ignore_index=True)

        # Calculate average predicted price per day
        avg_prices = future_data.groupby('date')['predicted_price'].mean().reset_index()
        
        # Create line chart
        chart = alt.Chart(avg_prices).mark_line().encode(
            x='date:T',
            y=alt.Y('predicted_price:Q', title='Average Predicted Price (€)'),
            tooltip=['date', 'predicted_price']
        ).properties(
            title=f'Average Predicted Price Trend for {selected_neighborhood}',
            width=700,
            height=400
        )

        st.altair_chart(chart, use_container_width=True)

        # Show top 5 most expensive days
        top_5_days = avg_prices.nlargest(5, 'predicted_price')
        st.subheader("Top 5 Most Expensive Days")
        st.table(top_5_days[['date', 'predicted_price']].set_index('date'))

        # Pagination for individual Airbnb predictions
        st.subheader("Individual Airbnb Predictions")
        items_per_page = 10
        num_pages = len(airbnb_ids) // items_per_page + (1 if len(airbnb_ids) % items_per_page > 0 else 0)
        page = st.selectbox("Page", range(1, num_pages + 1))

        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page

        for airbnb_id in airbnb_ids[start_idx:end_idx]:
            airbnb_data = future_data[future_data['airbnb_id'] == airbnb_id]
            st.write(f"Airbnb ID: {airbnb_id}")
            
            # Create line chart for individual Airbnb
            airbnb_chart = alt.Chart(airbnb_data).mark_line().encode(
                x='date:T',
                y=alt.Y('predicted_price:Q', title='Predicted Price (€)'),
                tooltip=['date', 'predicted_price']
            ).properties(
                title=f'Predicted Price Trend for Airbnb {airbnb_id}',
                width=600,
                height=200
            )
            
            st.altair_chart(airbnb_chart, use_container_width=True)

else:
    st.warning("Please select a valid neighborhood.")


