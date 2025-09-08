import streamlit as st
import pandas as pd
import pickle
import requests
import io

st.set_page_config(
    page_title="Swiggy Restaurant Recommendation",
    page_icon="https://cdn-icons-png.flaticon.com/512/3075/3075977.png",  # Example: food icon
    layout="wide"
)

@st.cache_data(show_spinner=False)
def load_csv_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    return pd.read_csv(url)

@st.cache_resource(show_spinner=False)
def load_pickle_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    return pickle.load(io.BytesIO(response.content))

file_ids = {
    "kmeans_trained_df": "1qgpaFP4o30QoUF2lkyPtL4kab5lb3-az",
    "clustered_df": "1gH-aMkgGXvauK56vlx-DfFXebuDsnLmN",
    "scaler": "1MnK1oVgqBidERQB1hNGVZbgTErqucHFk",
    "kmeans_model": "1upNugSoFIUTogVDioKEkHNoOWXvRD5OE",
    "city_encoder": "1e_eb84newGvAyDaZ6h_kRceFStwTg6af",
    "cuisine_encoder": "103K8wOgJWJYYRPce-KFVTqCV2tX0NrHI",
}

df = load_csv_from_drive(file_ids["kmeans_trained_df"])
clustered_df = load_csv_from_drive(file_ids["clustered_df"])

scaler = load_pickle_from_drive(file_ids["scaler"])
kmeans = load_pickle_from_drive(file_ids["kmeans_model"])
city_encoder = load_pickle_from_drive(file_ids["city_encoder"])
cuisine_encoder = load_pickle_from_drive(file_ids["cuisine_encoder"])

def recommend_by_all_inputs(city, cuisine, rating, rating_count, cost, df,
                            city_encoder, cuisine_encoder, scaler, kmeans, clustered_df):
    numeric_input = pd.DataFrame([[rating, rating_count, cost]],
                                 columns=['rating', 'rating_count', 'cost'])

    scaled_numeric = pd.DataFrame(scaler.transform(numeric_input), columns=numeric_input.columns)

    # Check if city and cuisine are known to encoder
    if city not in city_encoder.classes_:
        st.warning(f"City '{city}' not in known cities.")
        return pd.DataFrame()
    if cuisine not in cuisine_encoder.classes_:
        st.warning(f"Cuisine '{cuisine}' not in known cuisines.")
        return pd.DataFrame()

    city_encoded = pd.DataFrame(city_encoder.transform([[city]]),
                                columns=city_encoder.classes_)

    cuisine_encoded = pd.DataFrame(cuisine_encoder.transform([[cuisine]]),
                                   columns=cuisine_encoder.classes_)

    input_vector = pd.concat([scaled_numeric, city_encoded, cuisine_encoded], axis=1)

    # Add missing columns as zero
    missing_cols = set(df.columns) - set(input_vector.columns)
    for col in missing_cols:
        input_vector[col] = 0

    # Ensure same column order
    input_vector = input_vector[df.columns]

    # Debug print shapes
    st.write("Input vector shape:", input_vector.shape)
    st.write("Input vector columns:", input_vector.columns)

    cluster = kmeans.predict(input_vector)[0]

    return clustered_df[clustered_df['cluster'] == cluster]


def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Initialize screen
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'


# HOME SCREEN
if st.session_state['page'] == 'home':
    set_background("https://images.unsplash.com/photo-1517248135467-4c7edcad34c4")
    st.markdown("<h1 style='color:white;'>Welcome to Swiggy Restaurant Recommendation</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:white;'>Find the best places to eat, tailored to your taste!</h3>", unsafe_allow_html=True)

    if st.button("Go to Search"):
        st.session_state['page'] = 'search'
        st.rerun()


# SEARCH SCREEN
elif st.session_state['page'] == 'search':
    set_background("https://images.unsplash.com/photo-1600891964599-f61ba0e24092")

    # Inject custom CSS for white font color
    st.markdown(
        """
        <style>
        h2, label, .stTextInput label, .stNumberInput label {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h2 style='color:white;'>Search for Restaurants</h2>", unsafe_allow_html=True)

    city = st.text_input('Enter city')
    cuisine = st.text_input('Enter cuisine')
    rating = st.number_input('Enter rating', min_value=0.0, max_value=5.0, step=0.1)
    rating_count = st.number_input('Enter rating count', min_value=1)
    cost = st.number_input('Enter cost', min_value=1)

    if st.button("Search"):
        if city and cuisine and rating and rating_count and cost:
            st.session_state['inputs'] = {
                "city": city,
                "cuisine": cuisine,
                "rating": rating,
                "rating_count": rating_count,
                "cost": cost
            }
            st.session_state['page'] = 'results'
            st.rerun()
        else:
            st.warning("Please fill in all fields.")

    if st.button("Back to Home"):
        st.session_state['page'] = 'home'
        st.rerun()


# RESULTS SCREEN
elif st.session_state['page'] == 'results':
    set_background("https://images.unsplash.com/photo-1555396273-367ea4eb4db5")
    st.markdown("<h2 style='color:white;'>🍴 Recommended Restaurants</h2>", unsafe_allow_html=True)

    user_input = st.session_state.get('inputs', {})

    if not user_input:
        st.warning("No inputs provided. Please go back to Search.")
        if st.button("Back to Search"):
            st.session_state['page'] = 'search'
            st.rerun()
    else:
        result_df = recommend_by_all_inputs(
            user_input['city'], user_input['cuisine'], user_input['rating'],
            user_input['rating_count'], user_input['cost'],
            df, city_encoder, cuisine_encoder, scaler, kmeans, clustered_df
        )

        result_df = result_df.sort_values(by='rating', ascending=False)

        result_df = result_df[
            result_df['cuisine'].apply(lambda x: any(user_input['cuisine'].lower() in item.strip().lower()
                                                     for item in x.split(',')))]
        result_df = result_df[
            result_df['city'].apply(lambda x: any(user_input['city'].lower() in item.strip().lower()
                                                  for item in x.split(',')))]

        if result_df.empty:
            st.warning("No matching restaurants found for the given inputs.")
        else:
            st.dataframe(result_df[['name', 'city', 'cuisine', 'cost', 'rating', 'rating_count']].head(30).reset_index(drop=True))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Search Again"):
            st.session_state['page'] = 'search'
            st.rerun()
    with col2:
        if st.button("Back to Home"):
            st.session_state['page'] = 'home'
            st.rerun()

