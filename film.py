from datetime import date
import streamlit as st
import sqlite3
import pandas as pd
import random
# [DIHAPUS] Semua import dari scikit-surprise dihilangkan
# from surprise.model_selection import train_test_split # Untuk membagi data
# from surprise import accuracy # Untuk menghitung metrik akurasi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# [DIHAPUS] SVD, Dataset, Reader tidak lagi digunakan
# from surprise import SVD, Dataset, Reader
import numpy as np

# === STREAMLIT PAGE CONFIG (MUST BE FIRST) ===
st.set_page_config(page_title="🎬 Movie Recommendations", layout="centered")

# === Load data from database ===
@st.cache_data
def load_data():
    try:
        conn = sqlite3.connect("netflix.db")
        df = pd.read_sql_query("SELECT title, genre, year, description, poster_url FROM movies", conn)
        conn.close()
        
        df['genre'] = df['genre'].astype(str).str.strip().str.lower()
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

# === Content-Based Filtering (Tidak Berubah) ===
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
    if df.empty:
        return []
        
    genre = genre.lower().strip()
    
    filtered_df = df[
        (df["year"] == year) &
        (df["genre"].str.contains(genre, case=False, na=False))
    ].copy()
    
    if filtered_df.empty:
        return []

    if len(filtered_df) <= top_n:
        return filtered_df["title"].tolist()

    try:
        tfidf_subset = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix_subset = tfidf_subset.fit_transform(filtered_df["combined"])
        cosine_sim_subset = cosine_similarity(tfidf_matrix_subset, tfidf_matrix_subset)
        
        avg_scores = cosine_sim_subset.mean(axis=1)
        sorted_indices = np.argsort(avg_scores)[::-1]
        
        recommended_indices = sorted_indices[:top_n]
        return filtered_df.iloc[recommended_indices]["title"].tolist()
    except Exception:
        return filtered_df.sort_values('year', ascending=False)["title"].head(top_n).tolist()

# === Collaborative Filtering (Logic Baru) ===
@st.cache_data
def create_dummy_ratings():
    if df.empty:
        return pd.DataFrame()
        
    users = [f"user_{i}" for i in range(1, 11)]
    ratings = []
    
    for user in users:
        sampled_movies = df.sample(min(15, len(df)))["title"].tolist()
        for title in sampled_movies:
            rating = random.choices([3, 4, 5], weights=[1, 2, 2])[0]
            ratings.append([user, title, rating])
    
    return pd.DataFrame(ratings, columns=["user_id", "title", "rating"])

ratings_df = create_dummy_ratings()

# [DIUBAH] Fungsi untuk mempersiapkan model kolaboratif diganti total
@st.cache_data
def prepare_user_similarity_model(ratings):
    """Mempersiapkan user-item matrix dan user similarity matrix."""
    if ratings.empty:
        return None, None
    
    try:
        # Membuat user-item matrix
        user_item_matrix = ratings.pivot_table(index='user_id', columns='title', values='rating').fillna(0)
        
        # Menghitung kesamaan antar pengguna menggunakan cosine similarity
        user_similarity = cosine_similarity(user_item_matrix)
        
        # Mengubahnya menjadi DataFrame agar mudah dibaca
        user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
        
        return user_item_matrix, user_similarity_df
    except Exception as e:
        st.warning(f"Collaborative filtering unavailable: {e}")
        return None, None

# [BARU] Memanggil fungsi baru
user_item_matrix, user_similarity_df = prepare_user_similarity_model(ratings_df)


# [DIUBAH] Fungsi rekomendasi kolaboratif diganti total
def collaborative_filtering_recommendations(user_id, genre, year, top_n=20):
    """Rekomendasi Collaborative Filtering berdasarkan kesamaan pengguna."""
    if user_similarity_df is None or user_item_matrix is None:
        return []
    
    # 1. Temukan pengguna yang paling mirip
    if user_id not in user_similarity_df.columns:
        return [] # Pengguna baru atau tidak ada data
        
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:11] # Top 10 pengguna mirip

    # 2. Kumpulkan film yang disukai oleh pengguna-pengguna mirip tersebut
    similar_users_movies = user_item_matrix.loc[similar_users]
    recommended_movies = similar_users_movies.mean(axis=0).sort_values(ascending=False)

    # 3. Hapus film yang sudah ditonton oleh pengguna target
    movies_watched_by_target = user_item_matrix.loc[user_id]
    movies_to_recommend = recommended_movies.drop(movies_watched_by_target[movies_watched_by_target > 0].index, errors='ignore')

    # 4. Filter berdasarkan genre dan tahun yang diminta
    genre = genre.lower().strip()
    candidate_movies = df[
        (df["genre"].str.contains(genre, case=False, na=False)) &
        (df["year"] == year)
    ]
    
    final_recs = [movie for movie in movies_to_recommend.index if movie in candidate_movies['title'].values]
    
    return final_recs[:top_n]


# === Hybrid Recommendation (Sedikit Perubahan) ===
def hybrid_by_genre(user_id, genre, year, top_n=50):
    cb_recs = content_based_by_genre(genre, year, top_n=25)
    # [DIUBAH] Memanggil fungsi CF baru
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
    
    return df[df['title'].isin(hybrid)].head(top_n)

# === Get all movies by genre and year (Tidak berubah) ===
def get_all_movies_by_genre_and_year(genre, year):
    if df.empty:
        return pd.DataFrame()
    genre = genre.lower().strip()
    result = df[
        (df["genre"].str.contains(genre, case=False, na=False)) &
        (df["year"] == year)
    ]
    return result[['title', 'year', 'genre']].sort_values(by='title')

# === Extract unique genres for dropdown (Tidak berubah) ===
def get_unique_genres():
    if df.empty:
        return []
    unique_genres = df['genre'].dropna().unique().tolist()
    return sorted([genre for genre in unique_genres if genre and genre.strip()])

# === STREAMLIT UI (Tidak berubah) ===
st.title("🎥 Movie Recommendation System")
st.markdown("**Hybrid Filtering System** - Choose your preferences below:")

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

top_n = st.slider("📊 Number of recommendations to display", min_value=1, max_value=5, value=5)

tab1, tab2 = st.tabs(["🎯 Get Recommendations", "🎞️ Browse All Movies"])

with tab1:
    st.markdown("### 🎬 Personalized Movie Recommendations")
    if st.button("🔍 Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            recommended_df = hybrid_by_genre(user_input, genre_input, year_input, top_n)
            
            if not recommended_df.empty:
                st.success(f"🎉 Found **{len(recommended_df)}** recommendations:")
                columns = 5
                rows = (len(recommended_df) + columns - 1) // columns
                
                for row_idx in range(rows):
                    cols = st.columns(columns)
                    for col_idx in range(columns):
                        movie_idx = row_idx * columns + col_idx
                        if movie_idx < len(recommended_df):
                            movie = recommended_df.iloc[movie_idx]
                            with cols[col_idx]:
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
            for i, (_, row) in enumerate(all_movies.iterrows(), start=1):
                st.write(f"**{i}. {row['title']}** ({row['year']}) - *{row['genre'].title()}*")
        else:
            st.warning(f"🤷 No movies found for **{genre_input.title()}** genre from **{year_input}**.")
            available_years = df[df["genre"].str.contains(genre_input.lower(), case=False, na=False)]["year"].unique()
            if len(available_years) > 0:
                st.info(f"💡 **{genre_input.title()}** movies are available for years: {sorted(available_years, reverse=True)}")