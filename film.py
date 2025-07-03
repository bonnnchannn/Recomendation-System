from datetime import date
import streamlit as st
import sqlite3
import pandas as pd
import random
from surprise.model_selection import train_test_split # Untuk membagi data
from surprise import accuracy # Untuk menghitung metrik akurasi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import numpy as np

# === STREAMLIT PAGE CONFIG (MUST BE FIRST) ===
st.set_page_config(page_title="🎬 Movie Recommendations", layout="centered")

# === Load data from database ===
@st.cache_data
def load_data():
    try:
        conn = sqlite3.connect("netflix.db")
        # Di bagian load_data():
        df = pd.read_sql_query("SELECT title, genre, year, description, poster_url FROM movies", conn)
        conn.close()
        
        # Clean and process genre data
        df['genre'] = df['genre'].astype(str).str.strip().str.lower()
        # Keep original genre combinations intact (don't split by comma for genre filtering)
        # This preserves entries like "romance+comedy" as separate options
        
        # Handle missing descriptions
        df['description'] = df['description'].fillna('')
        df["combined"] = df["genre"] + " " + df["description"]
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("No data available. Please check your database connection.")
    st.stop()

# === Content-Based Filtering ===
@st.cache_data
def prepare_tfidf_matrix():
    if df.empty:
        return None, None
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["combined"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf, cosine_sim

tfidf, cosine_sim = prepare_tfidf_matrix()

def content_based_by_genre(genre, year, top_n=20):
    """
    Content-based filtering that handles both individual and combined genres
    """
    if df.empty:
        return []
        
    genre = genre.lower().strip()
    
    # Filter movies that exactly match the genre or contain it as part of combination
    filtered_df = df[
        (df["year"] == year) &
        (df["genre"].str.contains(genre, case=False, na=False))
    ].copy()
    
    if filtered_df.empty:
        return []

    # If we have few movies, return all of them
    if len(filtered_df) <= top_n:
        return filtered_df["title"].tolist()

    # Calculate similarity for filtered subset
    try:
        tfidf_subset = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix_subset = tfidf_subset.fit_transform(filtered_df["combined"])
        cosine_sim_subset = cosine_similarity(tfidf_matrix_subset, tfidf_matrix_subset)
        
        # Calculate average similarity scores
        avg_scores = cosine_sim_subset.mean(axis=1)
        
        # Get indices sorted by similarity
        sorted_indices = np.argsort(avg_scores)[::-1]
        
        recommended_indices = sorted_indices[:top_n]
        return filtered_df.iloc[recommended_indices]["title"].tolist()
    except Exception as e:
        # Fallback: return movies sorted by year
        return filtered_df.sort_values('year', ascending=False)["title"].head(top_n).tolist()

# === Collaborative Filtering with Dummy Data ===
@st.cache_data
def create_dummy_ratings():
    """Create dummy ratings data for collaborative filtering"""
    if df.empty:
        return pd.DataFrame()
        
    users = [f"user_{i}" for i in range(1, 11)]  # More users for better recommendations
    ratings = []
    
    # Ensure each user rates movies from different genres and years
    for user in users:
        # Sample movies from different years and genres
        sampled_movies = df.sample(min(15, len(df)))["title"].tolist()
        for title in sampled_movies:
            # Create more realistic ratings (slightly biased towards higher ratings)
            rating = random.choices([3, 4, 5], weights=[1, 2, 2])[0]
            ratings.append([user, title, rating])
    
    return pd.DataFrame(ratings, columns=["user_id", "title", "rating"])

ratings_df = create_dummy_ratings()

# Prepare collaborative filtering model
@st.cache_resource
def prepare_collaborative_model():
    if ratings_df.empty:
        return None
        
    try:
        reader = Reader(rating_scale=(1, 5))
        data_surprise = Dataset.load_from_df(ratings_df, reader)
        trainset, _ = train_test_split(data_surprise, test_size=0.2, random_state=42)
        
        algo = SVD(random_state=42)
        algo.fit(trainset)
        return algo
    except Exception as e:
        st.warning(f"Collaborative filtering unavailable: {e}")
        return None

algo = prepare_collaborative_model()

def collaborative_filtering_recommendations(user_id, genre, year, top_n=20):
    """
    Collaborative filtering that preserves original genre combinations
    """
    if algo is None or df.empty:
        return []
        
    genre = genre.lower().strip()
    
    # Get movies that match genre and year (using string contains for flexibility)
    genre_movies = df[
        (df["genre"].str.contains(genre, case=False, na=False)) &
        (df["year"] == year)
    ]["title"].tolist()
    
    if not genre_movies:
        return []
    
    predictions = []
    for movie in genre_movies:
        if movie in ratings_df['title'].unique():
            try:
                pred = algo.predict(user_id, movie)
                predictions.append((movie, pred.est))
            except Exception:
                continue
    
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    return [movie for movie, _ in predictions[:top_n]]

# === Hybrid Recommendation ===
def hybrid_by_genre(user_id, genre, year, top_n=50):
    cb_recs = content_based_by_genre(genre, year, top_n=25)
    cf_recs = collaborative_filtering_recommendations(user_id, genre, year, top_n=25)

    hybrid = []
    seen = set()
    max_len = max(len(cb_recs), len(cf_recs))
    
    for i in range(max_len):
        if i < len(cb_recs) and cb_recs[i] not in seen:
            hybrid.append(cb_recs[i])
            seen.add(cb_recs[i])
        if i < len(cf_recs) and cf_recs[i] not in seen:
            hybrid.append(cf_recs[i])
            seen.add(cf_recs[i])
    
    # Ambil data termasuk poster_url
    return df[df['title'].isin(hybrid)].head(top_n)

# === Get all movies by genre and year ===
def get_all_movies_by_genre_and_year(genre, year):
    """
    Get all movies that match the genre and year criteria
    Preserves original genre combinations
    """
    if df.empty:
        return pd.DataFrame()
        
    genre = genre.lower().strip()
    
    result = df[
        (df["genre"].str.contains(genre, case=False, na=False)) &
        (df["year"] == year)
    ]
    
    return result[['title', 'year', 'genre']].sort_values(by='title')

# === Extract unique genres for dropdown ===
def get_unique_genres():
    """Extract all unique genres from the dataset, preserving combinations"""
    if df.empty:
        return []
        
    # Get unique genre values directly (preserves combinations like "romance+comedy")
    unique_genres = df['genre'].dropna().unique().tolist()
    
    return sorted([genre for genre in unique_genres if genre and genre.strip()])

# === STREAMLIT UI ===
st.title("🎥 Movie Recommendation System")
st.markdown("**Hybrid Filtering System** - Choose your preferences below:")

# Create columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    users = [f"user_{i}" for i in range(1, 11)]
    user_input = st.selectbox("👤 Select User ID", users, index=0)

with col2:
    genre_list = get_unique_genres()
    if genre_list:
        genre_input = st.selectbox("🎭 Select Genre", genre_list)
    else:
        st.error("No genres available")
        st.stop()

with col3:
    year_list = sorted(df["year"].unique(), reverse=True)
    year_input = st.selectbox("📅 Select Release Year", year_list)

# Number of recommendations slider
top_n = st.slider("📊 Number of recommendations to display", min_value=1, max_value=5, value=5)

# Create tabs
tab1, tab2 = st.tabs(["🎯 Get Recommendations", "🎞️ Browse All Movies"])

with tab1:
    st.markdown("### 🎬 Personalized Movie Recommendations")
    if st.button("🔍 Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            recommended_df = hybrid_by_genre(user_input, genre_input, year_input, top_n)
            
            if not recommended_df.empty:
                st.success(f"🎉 Found **{len(recommended_df)}** recommendations:")

                # Calculate number of columns for display
                columns = 5  # Set the number of columns to 5
                rows = (len(recommended_df) // columns) + (1 if len(recommended_df) % columns != 0 else 0)
                
                for row_idx in range(rows):
                    cols = st.columns(columns)
                    for col_idx in range(columns):
                        movie_idx = row_idx * columns + col_idx
                        if movie_idx < len(recommended_df):
                            movie = recommended_df.iloc[movie_idx]
                            with cols[col_idx]:
                                # Show image and details for each movie
                                if pd.notna(movie['poster_url']) and movie['poster_url']:
                                    try:
                                        st.image(movie['poster_url'], width=150, caption=movie['title'])
                                    except Exception:
                                        st.write("🖼️ No Image")
                                else:
                                    st.write("🖼️ No Poster Available")
                                st.markdown(f"**{movie['title']}** ({movie['year']})")
                                st.markdown(f"*Genre: {movie['genre'].title()}*")
                                if movie['description']:
                                    st.markdown(movie['description'][:150] + "...")
                    st.markdown("---")
            else:
                st.warning(f"😔 No recommendations found for **{genre_input.title()}** movies from **{year_input}.")
    
with tab2:
    st.markdown("### 🔍 Browse All Movies by Genre & Year")
    
    if st.button("📋 Show All Movies", type="secondary"):
        all_movies = get_all_movies_by_genre_and_year(genre_input, year_input)
        
        if not all_movies.empty:
            st.success(f"📽️ Found **{len(all_movies)}** movies in **{genre_input.title()}** genre from **{year_input}**:")
            
            # Display in a more organized way
            for i, (_, row) in enumerate(all_movies.iterrows(), start=1):
                st.write(f"**{i}. {row['title']}** ({row['year']}) - *{row['genre'].title()}*")
        else:
            st.warning(f"🤷 No movies found for **{genre_input.title()}** genre from **{year_input}**.")
            
            # Show available years for this genre
            available_years = df[   
                df["genre"].str.contains(genre_input.lower(), case=False, na=False)
            ]["year"].unique()
            
            if len(available_years) > 0:
                st.info(f"💡 **{genre_input.title()}** movies are available for years: {sorted(available_years, reverse=True)}")
