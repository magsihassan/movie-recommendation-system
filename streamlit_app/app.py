import sys
import os
import streamlit as st
import pandas as pd
import random

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(current_dir)
sys.path.insert(0, ROOT_DIR)

try:
    from src.data_loader import load_movies
    from src.recommender import MovieRecommender
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Initialize recommender
@st.cache_resource
def load_recommender():
    return MovieRecommender()

recommender = load_recommender()

# Load movie data
@st.cache_data
def load_movie_data():
    return load_movies()

movies_df = load_movie_data()

# Streamlit UI with enhanced movie theme
st.set_page_config(
    page_title="CineAI - Smart Movie Recommendations",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="collapsed"
)

# Custom CSS for futuristic movie theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Exo+2:wght@300;400;500;600&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
        font-family: 'Exo 2', sans-serif;
    }
    
    .title-container {
        background: linear-gradient(90deg, #ff6b6b 0%, #4ecdc4 50%, #45b7d1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 4rem !important;
        font-weight: 900;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(78, 205, 196, 0.5);
        letter-spacing: 3px;
    }
    
    .subtitle {
        font-family: 'Exo 2', sans-serif;
        font-size: 1.5rem !important;
        font-weight: 300;
        color: #b0b0b0;
        margin-bottom: 2rem;
    }
    
    .model-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .model-card:hover {
        transform: translateY(-5px);
        border-color: #4ecdc4;
        box-shadow: 0 10px 30px rgba(78, 205, 196, 0.2);
    }
    
    .model-card.selected {
        border-color: #ff6b6b;
        background: rgba(255, 107, 107, 0.1);
    }
    
    .model-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .model-title {
        font-family: 'Orbitron', monospace;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #4ecdc4;
    }
    
    .model-desc {
        font-size: 0.9rem;
        color: #b0b0b0;
        line-height: 1.4;
    }
    
    .recommendation-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateX(5px);
        border-color: #45b7d1;
        box-shadow: 0 5px 20px rgba(69, 183, 209, 0.3);
    }
    
    .movie-title {
        font-family: 'Exo 2', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .movie-genres {
        font-size: 0.9rem;
        color: #4ecdc4;
        margin-bottom: 0.5rem;
    }
    
    .movie-rating {
        font-family: 'Orbitron', monospace;
        font-size: 1.1rem;
        color: #ffd700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.8rem 2rem;
        font-family: 'Exo 2', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(78, 205, 196, 0.4);
    }
    
    .section-header {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: #4ecdc4;
        margin: 2rem 0 1rem 0;
        text-align: center;
        text-shadow: 0 0 10px rgba(78, 205, 196, 0.3);
    }
    
    .stRadio>div {
        display: flex;
        gap: 1rem;
        justify-content: center;
    }
    
    .stRadio>div>label {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    .stRadio>div>label:hover {
        border-color: #4ecdc4 !important;
    }
    
    .stRadio>div>label[data-baseweb="radio"]>div:first-child {
        border-color: #4ecdc4 !important;
    }
    
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: #888;
        font-size: 0.9rem;
    }
    
    .neon-text {
        color: #4ecdc4;
        text-shadow: 0 0 10px #4ecdc4, 0 0 20px #4ecdc4, 0 0 30px #4ecdc4;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Header with animated title
st.markdown("""
<div class="title-container">
    <div class="main-title">🎬 CINEMATIC AI</div>
    <div class="subtitle">Your Personal Movie Discovery Engine Powered by Machine Learning</div>
</div>
""", unsafe_allow_html=True)

# Create columns for main content
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="section-header">🤖 SELECT AI MODEL</div>', unsafe_allow_html=True)
    
    # Model selection with enhanced cards
    model_option = st.radio(
        "",
        ["Content-Based Filtering", "Collaborative Filtering", "Hybrid Intelligence"],
        key="model_selector",
        label_visibility="collapsed"
    )
    
    # Model descriptions
    st.markdown("""
    <div style='margin-top: 2rem;'>
        <div class="model-card">
            <div class="model-icon">🎯</div>
            <div class="model-title">CONTENT-BASED AI</div>
            <div class="model-desc">Analyzes movie content, genres, and themes to find films with similar characteristics to your favorites.</div>
        </div>
        
        <div class="model-card">
            <div class="model-icon">👥</div>
            <div class="model-title">COLLABORATIVE AI</div>
            <div class="model-desc">Learns from millions of user ratings to predict what you'll love based on similar taste profiles.</div>
        </div>
        
        <div class="model-card">
            <div class="model-icon">🚀</div>
            <div class="model-title">HYBRID INTELLIGENCE</div>
            <div class="model-desc">Combines both approaches for the most accurate and diverse movie recommendations.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-header">🎭 YOUR MOVIE PREFERENCES</div>', unsafe_allow_html=True)
    
    # Dynamic input sections based on model selection
    if model_option in ["Content-Based Filtering", "Hybrid Intelligence"]:
        st.markdown("**🎥 SELECT MOVIES YOU LOVE**")
        movie_titles = movies_df["title"].astype(str).tolist()
        selected_movies = st.multiselect(
            "Choose movies that match your taste:",
            movie_titles,
            placeholder="Search for movies...",
            max_selections=5,
            label_visibility="collapsed"
        )
        
        # Show movie posters concept
        if selected_movies:
            st.markdown("**Your Selection:**")
            cols = st.columns(len(selected_movies))
            for idx, (col, movie) in enumerate(zip(cols, selected_movies)):
                with col:
                    # Generate a random "poster" color for visual appeal
                    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
                    color = colors[idx % len(colors)]
                    st.markdown(f"""
                    <div style='
                        background: {color};
                        border-radius: 10px;
                        padding: 1rem;
                        text-align: center;
                        color: white;
                        font-weight: bold;
                        margin: 0.2rem;
                        min-height: 80px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    '>
                        {movie[:25]}{'...' if len(movie) > 25 else ''}
                    </div>
                    """, unsafe_allow_html=True)

    if model_option in ["Collaborative Filtering", "Hybrid Intelligence"]:
        st.markdown("**👤 YOUR PROFILE**")
        user_id = st.slider(
            "Select Your User Profile:",
            min_value=1,
            max_value=6040,
            value=random.randint(1, 100),
            help="Different profiles have different movie taste patterns"
        )
        
        # Show user's taste profile
        try:
            user_ratings = recommender.ratings_df[recommender.ratings_df['userId'] == user_id]
            if not user_ratings.empty:
                avg_rating = user_ratings['rating'].mean()
                rating_count = len(user_ratings)
                st.markdown(f"""
                <div style='
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 10px;
                    padding: 1rem;
                    margin: 1rem 0;
                '>
                    <div style='color: #4ecdc4; font-weight: bold;'>Profile Analysis:</div>
                    <div>🎯 Average Rating: <span style='color: #ffd700;'>{avg_rating:.1f}/5</span></div>
                    <div>📊 Movies Rated: <span style='color: #4ecdc4;'>{rating_count}</span></div>
                </div>
                """, unsafe_allow_html=True)
        except:
            pass

    # Alpha slider for hybrid model
    if model_option == "Hybrid Intelligence":
        st.markdown("**⚖️ AI BALANCE**")
        alpha = st.slider(
            "Content vs Collaborative Intelligence:",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Adjust how much weight to give to content analysis vs collaborative patterns"
        )
        
        # Visual indicator
        content_width = int(alpha * 100)
        collab_width = 100 - content_width
        
        st.markdown(f"""
        <div style='
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            height: 20px;
            margin: 1rem 0;
            overflow: hidden;
        '>
            <div style='
                background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
                height: 100%;
                width: 100%;
                display: flex;
            '>
                <div style='
                    background: #ff6b6b;
                    height: 100%;
                    width: {content_width}%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 0.7rem;
                    font-weight: bold;
                '>Content {content_width}%</div>
                <div style='
                    background: #4ecdc4;
                    height: 100%;
                    width: {collab_width}%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 0.7rem;
                    font-weight: bold;
                '>Collaborative {collab_width}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Recommendation button with enhanced styling
st.markdown("""
<div style='text-align: center; margin: 2rem 0;'>
    <div class="pulse">
""", unsafe_allow_html=True)

if st.button("🚀 GENERATE SMART RECOMMENDATIONS", key="recommend_btn"):
    st.markdown("""
    </div>
    """, unsafe_allow_html=True)
    
    # Loading animation
    with st.spinner("""
        <div style='text-align: center;'>
            <div style='color: #4ecdc4; font-size: 1.2rem; margin-bottom: 1rem;'>🧠 AI is analyzing patterns...</div>
            <div style='color: #b0b0b0;'>Scanning 1 million ratings and movie metadata</div>
        </div>
    """):
        try:
            if model_option == "Content-Based Filtering":
                if not selected_movies:
                    st.warning("🎬 Please select at least one movie to get content-based recommendations.")
                else:
                    movie_id = movies_df[movies_df["title"] == selected_movies[0]]["movieId"].iloc[0]
                    recommendations = recommender.content_recommend(movie_id, k=10)
                    
                    if not recommendations.empty:
                        st.markdown('<div class="section-header">🎯 CONTENT-BASED RECOMMENDATIONS</div>', unsafe_allow_html=True)
                        st.markdown(f"<div style='text-align: center; color: #b0b0b0; margin-bottom: 2rem;'>Movies similar to <span class='neon-text'>{selected_movies[0]}</span></div>", unsafe_allow_html=True)
                        
                        for idx, row in recommendations.iterrows():
                            pred_rating = row.get('pred_rating', 4.0)
                            similarity = row.get('similarity_score', 0)
                            
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <div class="movie-title">{row['title']}</div>
                                <div class="movie-genres">{row['genres']}</div>
                                <div style='display: flex; justify-content: space-between; align-items: center;'>
                                    <div class="movie-rating">
                                        ⭐ {pred_rating:.2f}/5.00
                                    </div>
                                    <div style='color: #ff6b6b; font-size: 0.9rem;'>
                                        🔍 Match: {similarity:.1%}
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            
            elif model_option == "Collaborative Filtering":
                recommendations = recommender.collaborative_recommend(user_id, k=10)
                
                if not recommendations.empty:
                    st.markdown('<div class="section-header">👥 PERSONALIZED RECOMMENDATIONS</div>', unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; color: #b0b0b0; margin-bottom: 2rem;'>Curated for User Profile <span class='neon-text'>#{user_id}</span></div>", unsafe_allow_html=True)
                    
                    for idx, row in recommendations.iterrows():
                        pred_rating = row.get('pred_rating', 4.0)
                        
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <div class="movie-title">{row['title']}</div>
                            <div class="movie-genres">{row['genres']}</div>
                            <div class="movie-rating">
                                ⭐ {pred_rating:.2f}/5.00
                                <span style='color: #4ecdc4; font-size: 0.8rem; margin-left: 1rem;'>
                                    🤖 AI Confidence: {(pred_rating/5*100):.0f}%
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            else:  # Hybrid Intelligence
                if not selected_movies:
                    st.warning("🎬 Please select at least one movie for hybrid recommendations.")
                else:
                    user_movie_ids = movies_df[movies_df["title"].isin(selected_movies)]["movieId"].tolist()
                    recommendations = recommender.hybrid_recommend(user_id, user_movie_ids, top_n=10, alpha=alpha)
                    
                    if not recommendations.empty:
                        st.markdown('<div class="section-header">🚀 HYBRID INTELLIGENCE RECOMMENDATIONS</div>', unsafe_allow_html=True)
                        st.markdown(f"<div style='text-align: center; color: #b0b0b0; margin-bottom: 2rem;'>Powered by advanced AI combining multiple algorithms</div>", unsafe_allow_html=True)
                        
                        for idx, row in recommendations.iterrows():
                            pred_rating = row.get('pred_rating', 4.0)
                            
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <div class="movie-title">{row['title']}</div>
                                <div class="movie-genres">{row['genres']}</div>
                                <div style='display: flex; justify-content: space-between; align-items: center;'>
                                    <div class="movie-rating">
                                        ⭐ {pred_rating:.2f}/5.00
                                    </div>
                                    <div style='display: flex; gap: 1rem;'>
                                        <span style='color: #ff6b6b; font-size: 0.8rem;'>🎯 Content</span>
                                        <span style='color: #4ecdc4; font-size: 0.8rem;'>👥 Collaborative</span>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Error generating recommendations: {str(e)}")

else:
    st.markdown("""
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <div style='font-size: 1.1rem; margin-bottom: 1rem;'>
        <span class="neon-text">CINEMATIC AI</span> • Your Personal Movie Discovery Engine
    </div>
    <div style='color: #666;'>
        Powered by TF-IDF • SVD • Hybrid Machine Learning Models<br>
        Trained on 1,000,000+ ratings from MovieLens 1M Dataset<br>
        🎬 Lights, Camera, AI Action! 🍿
    </div>
</div>
""", unsafe_allow_html=True)

# Add some movie quotes for fun
movie_quotes = [
    "“May the Force be with your movie choices.”",
    "“Here's looking at you, movie lover.”", 
    "“You're gonna need a bigger watchlist.”",
    "“I'll be back with more recommendations.”",
    "“You had me at hello movie night.”"
]

st.markdown(f"""
<div style='
    text-align: center;
    color: #666;
    font-style: italic;
    margin-top: 2rem;
    padding: 1rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
'>
    {random.choice(movie_quotes)}
</div>
""", unsafe_allow_html=True)