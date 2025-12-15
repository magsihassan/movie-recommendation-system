import sys
import os
import random
import requests
import streamlit as st
import pandas as pd
import re

# -------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(current_dir)
sys.path.insert(0, ROOT_DIR)

try:
    from src.data_loader import load_movies
    from src.recommender import MovieRecommender
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# -------------------------------------------------------------------
# LOAD MODELS + DATA
# -------------------------------------------------------------------
@st.cache_resource
def load_recommender():
    return MovieRecommender()

recommender = load_recommender()

@st.cache_data
def load_movie_data():
    return load_movies()

movies_df = load_movie_data()

# -------------------------------------------------------------------
# TMDB POSTER HELPER (OPTIONAL)
# -------------------------------------------------------------------
# API key is loaded from .streamlit/secrets.toml or environment variable
# See .streamlit/secrets.toml.example for setup instructions
TMDB_API_KEY = None

try:
    TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", None)
except Exception:
    pass

if not TMDB_API_KEY:
    TMDB_API_KEY = os.getenv("TMDB_API_KEY", None)

# -----------------------------------------------------------
# 🔥 TMDB TITLE NORMALIZATION + BEST-MATCH POSTER FETCH
# -----------------------------------------------------------

def normalize_title(movie_title: str):
    """
    MovieLens → TMDB title cleanup
    Handles:
    - Removing (YEAR)
    - Converting 'Title, The' → 'The Title'
    - Extracting release year
    """
    title = movie_title.strip()

    # Extract year
    year_match = re.search(r"\((\d{4})\)", title)
    year = int(year_match.group(1)) if year_match else None

    # Remove (YEAR)
    title = re.sub(r"\(\d{4}\)", "", title).strip()

    # Handle ", The" / ", A" / ", An"
    if ", The" in title:
        title = "The " + title.replace(", The", "")
    if ", A" in title:
        title = "A " + title.replace(", A", "")
    if ", An" in title:
        title = "An " + title.replace(", An", "")

    return title.strip(), year


@st.cache_data(show_spinner=False)
def get_poster_url(movie_title: str):
    """Return best-match TMDB poster URL for a MovieLens movie."""
    if not TMDB_API_KEY:
        return None

    clean_title, year = normalize_title(movie_title)

    try:
        params = {"api_key": TMDB_API_KEY, "query": clean_title}
        if year:
            params["year"] = year  # improves accuracy

        resp = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params=params,
            timeout=6,
        )
        resp.raise_for_status()

        results = resp.json().get("results", [])
        if not results:
            return None

        # Pick best match by vote_count + popularity
        results = sorted(
            results,
            key=lambda x: (x.get("vote_count", 0), x.get("popularity", 0)),
            reverse=True,
        )
        best = results[0]

        poster_path = best.get("poster_path")
        if not poster_path:
            return None

        return f"https://image.tmdb.org/t/p/w342{poster_path}"

    except Exception:
        return None

@st.cache_data(show_spinner=False)
def get_movie_details(movie_title: str):
    """Fetch movie details including trailer from TMDB."""
    if not TMDB_API_KEY:
        return None
    
    clean_title, year = normalize_title(movie_title)
    
    try:
        # Search for movie
        params = {"api_key": TMDB_API_KEY, "query": clean_title}
        if year:
            params["year"] = year
        
        resp = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params=params,
            timeout=6,
        )
        resp.raise_for_status()
        
        results = resp.json().get("results", [])
        if not results:
            return None
        
        # Get best match
        results = sorted(
            results,
            key=lambda x: (x.get("vote_count", 0), x.get("popularity", 0)),
            reverse=True,
        )
        movie = results[0]
        movie_id = movie.get("id")
        
        # Fetch movie details with videos
        details_resp = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": TMDB_API_KEY, "append_to_response": "videos,credits"},
            timeout=6,
        )
        details_resp.raise_for_status()
        details = details_resp.json()
        
        # Extract trailer (prefer YouTube trailers)
        trailer_key = None
        videos = details.get("videos", {}).get("results", [])
        for video in videos:
            if video.get("site") == "YouTube" and video.get("type") == "Trailer":
                trailer_key = video.get("key")
                break
        
        # Get top cast
        cast = details.get("credits", {}).get("cast", [])[:5]
        cast_names = [c.get("name") for c in cast if c.get("name")]
        
        return {
            "id": movie_id,
            "title": details.get("title", movie_title),
            "overview": details.get("overview", "No description available."),
            "poster_path": details.get("poster_path"),
            "backdrop_path": details.get("backdrop_path"),
            "release_date": details.get("release_date", ""),
            "runtime": details.get("runtime", 0),
            "vote_average": details.get("vote_average", 0),
            "vote_count": details.get("vote_count", 0),
            "genres": [g.get("name") for g in details.get("genres", [])],
            "trailer_key": trailer_key,
            "cast": cast_names,
        }
    except Exception:
        return None

# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Cinematic AI",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------------------------
# THEME STATE MANAGEMENT
# -------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "selected_movie_for_details" not in st.session_state:
    st.session_state.selected_movie_for_details = None

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

is_dark = st.session_state.theme == "dark"

# -------------------------------------------------------------------
# GLOBAL CSS  (with theme support, animations, skeleton, modal)
# -------------------------------------------------------------------
# Theme colors
if is_dark:
    theme_bg = "radial-gradient(circle at top, #1a1a2e 0, #0b0c10 45%, #000 100%)"
    theme_text = "#ffffff"
    theme_card_bg = "rgba(16, 24, 40, 0.9)"
    theme_border = "rgba(148,163,184,0.4)"
    theme_subtitle = "#b0b0b0"
    theme_secondary = "#9ca3af"
else:
    theme_bg = "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)"
    theme_text = "#1a1a2e"
    theme_card_bg = "rgba(255, 255, 255, 0.95)"
    theme_border = "rgba(0,0,0,0.15)"
    theme_subtitle = "#4b5563"
    theme_secondary = "#6b7280"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Exo+2:wght@300;400;600&display=swap');

body {{
    background: {theme_bg};
    color: {theme_text};
    font-family: 'Exo 2', sans-serif;
}}

.app-container {{
    max-width: 1200px;
    margin: 0 auto;
}}

/* MAIN HEADER */
.neon-header {{
    font-family: 'Orbitron', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    color: #4ecdc4;
    text-shadow:
        0 0 8px rgba(78,205,196,0.7),
        0 0 18px rgba(78,205,196,0.9),
        0 0 30px rgba(78,205,196,0.7);
    letter-spacing: 2px;
    animation: hdr-pulse 3s ease-in-out infinite alternate;
}}


.neon-subtitle {{
    color: {theme_subtitle};
    font-size: 1.1rem;
}}

@keyframes hdr-pulse {{
    0%% {{ text-shadow: 0 0 6px rgba(78,205,196,0.5); }}
    100%% {{ text-shadow: 0 0 24px rgba(78,205,196,0.9); }}
}}

/* SECTION HEADER */
.section-header {{
    font-family: 'Orbitron', sans-serif;
    font-size: 1.7rem;
    color: #4ecdc4;
    text-align: center;
    margin: 1.5rem 0 1rem 0;
    text-shadow: 0 0 8px rgba(78,205,196,0.8);
}}

/* MODEL CARDS */
.model-card {{
    background: {theme_card_bg};
    border-radius: 18px;
    border: 1px solid {theme_border};
    padding: 1.2rem 1.3rem;
    margin-bottom: 1.1rem;
    backdrop-filter: blur(14px);
    box-shadow:
        0 0 0 1px rgba(15,23,42,0.7),
        0 12px 35px rgba(15,23,42,0.95);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    transform-style: preserve-3d;
    transform-origin: center;
}}

.model-card:hover {{
    transform: translateY(-6px) rotate3d(1, -1, 0, 6deg);
    box-shadow:
        0 0 0 1px rgba(94,234,212,0.4),
        0 18px 55px rgba(56,189,248,0.55);
    border-color: #4ecdc4;
    cursor: pointer;
}}

.model-card.selected {{
    border-color: #ff6b6b;
    box-shadow:
        0 0 0 1px rgba(248,113,113,0.6),
        0 18px 60px rgba(248,113,113,0.85);
    animation: neon-pulse 2.2s ease-in-out infinite alternate;
}}

@keyframes neon-pulse {{
    0%% {{
        box-shadow:
            0 0 0 1px rgba(248,113,113,0.4),
            0 14px 40px rgba(248,113,113,0.5);
    }}
    100%% {{
        box-shadow:
            0 0 0 2px rgba(248,113,113,0.7),
            0 18px 64px rgba(248,113,113,0.9);
    }}
}}

.model-icon {{
    font-size: 2.4rem;
    margin-bottom: 0.4rem;
}}

.model-title {{
    font-family: 'Orbitron';
    font-size: 1.3rem;
    font-weight: 700;
    color: #e5e7eb;
    margin-bottom: 0.2rem;
}}

.model-desc {{
    font-size: 0.9rem;
    color: {theme_secondary};
}}

/* HORIZONTAL RECOMMENDATION ROW */
.rec-row-wrapper {{
    position: relative;
    margin-top: 1rem;
}}

.rec-row {{
    display: flex;
    overflow-x: auto;
    gap: 1rem;
    padding-bottom: 0.5rem;
    scroll-behavior: smooth;
}}

.rec-row::-webkit-scrollbar {{
    height: 8px;
}}
.rec-row::-webkit-scrollbar-track {{
    background: rgba(15,23,42,0.9);
}}
.rec-row::-webkit-scrollbar-thumb {{
    background: linear-gradient(90deg, #4ecdc4, #ff6b6b);
    border-radius: 999px;
}}

.rec-poster {{
    width: 100%%;
    height: 800px;
    background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
    overflow: hidden;
}}

.rec-poster img {{
    width: 100%%;
    height: 100%%;
    object-fit: cover;
    display: block;
}}

.rec-body {{
    padding: 0.7rem 0.75rem 0.8rem 0.75rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%%;
}}

.movie-title {{
    font-size: 0.95rem;
    font-weight: 600;
    color: #e5e7eb;
    margin-bottom: 0.2rem;
}}

.movie-genres {{
    font-size: 0.78rem;
    color: #67e8f9;
    margin-bottom: 0.3rem;
}}

.movie-rating {{
    font-family: 'Orbitron';
    font-size: 0.85rem;
    color: #facc15;
}}

.movie-meta-sub {{
    font-size: 0.7rem;
    color: {theme_secondary};
    text-align: right;
}}

.pill {{
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 0.1rem 0.5rem;
    font-size: 0.7rem;
    background: rgba(15,23,42,0.9);
    color: #e5e7eb;
    border: 1px solid rgba(148,163,184,0.6);
}}

/* THEME TOGGLE BUTTON */
.theme-toggle {{
    position: fixed;
    top: 70px;
    right: 20px;
    z-index: 1000;
    background: {theme_card_bg};
    border: 1px solid {theme_border};
    border-radius: 50%;
    width: 45px;
    height: 45px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 1.4rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}}

.theme-toggle:hover {{
    transform: scale(1.1);
    box-shadow: 0 6px 25px rgba(78,205,196,0.4);
}}

/* SKELETON LOADING */
.skeleton {{
    background: linear-gradient(90deg, 
        {theme_card_bg} 25%, 
        rgba(78,205,196,0.2) 50%, 
        {theme_card_bg} 75%);
    background-size: 200% 100%;
    animation: skeleton-shimmer 1.5s infinite;
    border-radius: 12px;
}}

.skeleton-card {{
    width: 220px;
    height: 380px;
    flex: 0 0 220px;
    border-radius: 16px;
}}

.skeleton-poster {{
    height: 280px;
    border-radius: 16px 16px 0 0;
}}

.skeleton-text {{
    height: 16px;
    margin: 8px 12px;
    border-radius: 4px;
}}

.skeleton-text-short {{
    width: 60%;
}}

@keyframes skeleton-shimmer {{
    0% {{ background-position: 200% 0; }}
    100% {{ background-position: -200% 0; }}
}}

/* CARD FADE-IN ANIMATION */
.rec-card {{
    flex: 0 0 300px;
    max-width: 220px;
    height: 500px;  
    background: radial-gradient(circle at top left, rgba(56,189,248,0.16), rgba(15,23,42,0.96));
    border-radius: 16px;
    border: 1px solid {theme_border};
    overflow: hidden;
    box-shadow: 0 14px 40px rgba(15,23,42,0.9);
    backdrop-filter: blur(10px);
    display: flex;
    flex-direction: column;
    animation: card-fade-in 0.5s ease-out forwards;
    opacity: 0;
    transform: translateY(20px);
    cursor: pointer;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}}

.rec-card:hover {{
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 20px 50px rgba(78,205,196,0.3);
}}

@keyframes card-fade-in {{
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

/* MODAL OVERLAY */
.modal-overlay {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.85);
    backdrop-filter: blur(8px);
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: modal-fade-in 0.3s ease;
}}

@keyframes modal-fade-in {{
    from {{ opacity: 0; }}
    to {{ opacity: 1; }}
}}

.modal-content {{
    background: {theme_card_bg};
    border-radius: 24px;
    border: 1px solid {theme_border};
    max-width: 900px;
    width: 90%;
    max-height: 90vh;
    overflow-y: auto;
    box-shadow: 0 25px 80px rgba(0,0,0,0.5);
    animation: modal-slide-up 0.4s ease;
}}

@keyframes modal-slide-up {{
    from {{ transform: translateY(50px); opacity: 0; }}
    to {{ transform: translateY(0); opacity: 1; }}
}}

.modal-header {{
    position: relative;
    height: 300px;
    overflow: hidden;
    border-radius: 24px 24px 0 0;
}}

.modal-backdrop {{
    width: 100%;
    height: 100%;
    object-fit: cover;
}}

.modal-close {{
    position: absolute;
    top: 15px;
    right: 15px;
    background: rgba(0,0,0,0.7);
    border: none;
    color: white;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    font-size: 1.5rem;
    cursor: pointer;
    transition: all 0.2s ease;
    z-index: 10;
}}

.modal-close:hover {{
    background: #ff6b6b;
    transform: scale(1.1);
}}

.modal-body {{
    padding: 1.5rem 2rem 2rem 2rem;
}}

.modal-title {{
    font-family: 'Orbitron', sans-serif;
    font-size: 1.8rem;
    color: #4ecdc4;
    margin-bottom: 0.5rem;
}}

.modal-meta {{
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}}

.modal-meta-item {{
    display: flex;
    align-items: center;
    gap: 0.3rem;
    color: {theme_secondary};
    font-size: 0.9rem;
}}

.modal-overview {{
    color: {theme_text};
    line-height: 1.6;
    margin-bottom: 1.5rem;
}}

.modal-cast {{
    color: {theme_secondary};
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}}

.modal-trailer {{
    border-radius: 12px;
    overflow: hidden;
    margin-top: 1rem;
}}

.modal-trailer iframe {{
    width: 100%;
    height: 400px;
    border: none;
}}

/* PROGRESS BAR */
.progress-container {{
    width: 100%;
    height: 6px;
    background: rgba(15,23,42,0.5);
    border-radius: 999px;
    overflow: hidden;
    margin: 1rem 0;
}}

.progress-bar {{
    height: 100%;
    background: linear-gradient(90deg, #4ecdc4, #ff6b6b, #4ecdc4);
    background-size: 200% 100%;
    animation: progress-animate 1.5s linear infinite;
    border-radius: 999px;
}}

@keyframes progress-animate {{
    0% {{ background-position: 0% 0; }}
    100% {{ background-position: 200% 0; }}
}}

/* MODE TRANSITION */
.stApp {{
    transition: background 0.5s ease;
}}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------
# Theme toggle button
theme_icon = "☀️" if is_dark else "🌙"
col_header1, col_header2, col_header3 = st.columns([1, 4, 1])
with col_header3:
    if st.button(theme_icon, key="theme_toggle", help=f"Switch to {'light' if is_dark else 'dark'} mode"):
        toggle_theme()
        st.rerun()

st.markdown(
    """
<div class="app-container" style="text-align:center; margin-bottom: 1.5rem;">
    <div class="neon-header">CINEMATIC AI</div>
    <div class="neon-subtitle">
        Hyper-visual Movie Discovery Powered by Hybrid Machine Learning
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# LAYOUT
# -------------------------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(
        '<div class="section-header">🤖 SELECT AI MODEL</div>',
        unsafe_allow_html=True,
    )

    MODEL_OPTIONS = [
        "Content-Based Filtering",
        "Collaborative Filtering",
        "Hybrid Intelligence",
    ]

    if "model_selector" not in st.session_state:
        st.session_state.model_selector = MODEL_OPTIONS[0]

    def render_model_card(model_name: str, icon: str, title: str, desc: str):
        is_selected = st.session_state.model_selector == model_name
        css = "model-card selected" if is_selected else "model-card"

        # Button controls state (hidden label text)
        clicked = st.button(title, key=f"btn_{model_name}")
        if clicked:
            st.session_state.model_selector = model_name

        st.markdown(
            f'<div class="{css}">'
            f'<div class="model-icon">{icon}</div>'
            f'<div class="model-title">{title}</div>'
            f'<div class="model-desc">{desc}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    render_model_card(
        "Content-Based Filtering",
        "🎯",
        "CONTENT-BASED AI",
        "Find movies similar to your favorites using content, genres, and themes.",
    )
    render_model_card(
        "Collaborative Filtering",
        "👥",
        "COLLABORATIVE AI",
        "Predict what you'll love based on users with similar rating patterns.",
    )
    render_model_card(
        "Hybrid Intelligence",
        "🚀",
        "HYBRID INTELLIGENCE",
        "Fuse content similarity and collaborative patterns for maximum accuracy.",
    )

    model_option = st.session_state.model_selector

# -------------------------------------------------------------------
# RIGHT COLUMN – INPUTS & SELECTION GALLERY
# -------------------------------------------------------------------
with col2:
    st.markdown(
        '<div class="section-header">🎭 YOUR MOVIE PREFERENCES</div>',
        unsafe_allow_html=True,
    )

    # Movies selection
    if model_option in ["Content-Based Filtering", "Hybrid Intelligence"]:
        movie_titles = movies_df["title"].astype(str).tolist()
        selected_movies = st.multiselect(
            "🎥 Movies you already enjoy:",
            movie_titles,
            max_selections=5,
            placeholder="Search movies...",
        )
    else:
        selected_movies = None

    # User ID
    if model_option in ["Collaborative Filtering", "Hybrid Intelligence"]:
        user_id = st.number_input(
            "👤 User ID (MovieLens):",
            min_value=1,
            max_value=6040,
            value=1,
            step=1,
        )
    else:
        user_id = None

    # Hybrid alpha
    if model_option == "Hybrid Intelligence":
        alpha = st.slider(
            "⚖️ Hybrid balance (Content ↔ Collaborative)",
            0.0,
            1.0,
            0.6,
            0.1,
        )
    else:
        alpha = None

    # Genre filtering
    st.markdown("---")
    all_genres = set()
    for genres in movies_df["genres"].dropna():
        for g in str(genres).split("|"):
            if g.strip():
                all_genres.add(g.strip())
    all_genres = sorted(all_genres)
    
    selected_genres = st.multiselect(
        "🎭 Filter by Genres (optional):",
        all_genres,
        placeholder="Select genres to filter...",
        help="Recommendations will be filtered to include only movies with these genres"
    )

    # Selection gallery
    if selected_movies:
        st.markdown("**Your selection gallery**")
        n = len(selected_movies)
        cols_sel = st.columns(min(n, 4))
        for idx, title in enumerate(selected_movies):
            col = cols_sel[idx % len(cols_sel)]
            with col:
                poster_url = get_poster_url(title)
                if poster_url:
                    # FIX: show full poster, scaled down, not full-width
                    st.image(poster_url, width=220)
                else:
                    colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#facc15", "#a855f7"]
                    color = colors[idx % len(colors)]
                    truncated = title[:22] + ("…" if len(title) > 22 else "")
                    col_html = (
                        "<div style="
                        f'"background:{color};'
                        'border-radius:12px;'
                        'height:140px;'
                        'display:flex;'
                        'align-items:center;'
                        'justify-content:center;'
                        'color:white;'
                        'font-weight:600;'
                        'text-align:center;'
                        'padding:0.5rem;">'
                        f"{truncated}"
                        "</div>"
                    )
                    st.markdown(col_html, unsafe_allow_html=True)

# -------------------------------------------------------------------
# GENERATE RECOMMENDATIONS
# -------------------------------------------------------------------
st.markdown(
    "<div style='text-align:center;margin-top:2rem;'>",
    unsafe_allow_html=True,
)

# Store recommendations in session state for persistence
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "rec_title_prefix" not in st.session_state:
    st.session_state.rec_title_prefix = ""

if st.button("🚀 Generate Smart Recommendations", key="generate_btn"):
    with st.spinner("🧠 Letting the AI binge-watch on your behalf..."):
        recs = None
        title_prefix = ""

        try:
            if model_option == "Content-Based Filtering":
                title_prefix = "🎯 Content-Based Recommendations"
                if not selected_movies:
                    st.warning("Please select at least one movie.")
                else:
                    movie_id = movies_df.loc[
                        movies_df["title"] == selected_movies[0], "movieId"
                    ].iloc[0]
                    recs = recommender.content_recommend(movie_id, k=10)

            elif model_option == "Collaborative Filtering":
                title_prefix = "👥 Personalized Recommendations"
                if not user_id:
                    st.warning("Please enter a valid user ID.")
                else:
                    recs = recommender.collaborative_recommend(int(user_id), k=10)

            else:  # Hybrid
                title_prefix = "🚀 Hybrid AI Recommendations"
                if not selected_movies or not user_id:
                    st.warning(
                        "Select at least one movie and provide a user ID for hybrid mode."
                    )
                else:
                    ids = movies_df[movies_df["title"].isin(selected_movies)][
                        "movieId"
                    ].tolist()
                    recs = recommender.hybrid_recommend(
                        int(user_id), ids, top_n=10, alpha=alpha
                    )

            if recs is not None and not recs.empty:
                # Apply genre filter if genres are selected
                if selected_genres:
                    def has_genre(movie_genres, filter_genres):
                        movie_genre_set = set(str(movie_genres).split("|"))
                        return any(g in movie_genre_set for g in filter_genres)
                    
                    recs = recs[recs["genres"].apply(lambda x: has_genre(x, selected_genres))]
                    
                    if recs.empty:
                        st.warning(f"No recommendations found matching genres: {', '.join(selected_genres)}. Try removing some genre filters.")
                
                # Store in session state
                st.session_state.recommendations = recs.reset_index(drop=True)
                st.session_state.rec_title_prefix = title_prefix
            else:
                st.info("No recommendations to display yet. Try adjusting your inputs.")
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")

# -------------------------------------------------------------------
# DISPLAY RECOMMENDATIONS IN GRID
# -------------------------------------------------------------------
if st.session_state.recommendations is not None and not st.session_state.recommendations.empty:
    recs = st.session_state.recommendations
    title_prefix = st.session_state.rec_title_prefix
    
    st.markdown(
        f"<h2 style='color:#4ecdc4; text-align:center;'>{title_prefix}</h2>",
        unsafe_allow_html=True,
    )
    
    # Grid layout - 5 columns
    num_cols = 5
    rec_list = recs.head(10).to_dict('records')
    
    for row_start in range(0, len(rec_list), num_cols):
        cols = st.columns(num_cols)
        for col_idx, col in enumerate(cols):
            rec_idx = row_start + col_idx
            if rec_idx < len(rec_list):
                movie = rec_list[rec_idx]
                title = str(movie["title"])
                genres = str(movie["genres"])
                rating = float(movie.get("pred_rating", 4.0))
                poster_url = get_poster_url(title)
                
                with col:
                    # Movie card with poster
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.markdown(
                            "<div style='height:200px;background:linear-gradient(135deg,#ff6b6b,#4ecdc4);"
                            "border-radius:12px;display:flex;align-items:center;justify-content:center;"
                            "font-size:3rem;'>🎬</div>",
                            unsafe_allow_html=True
                        )
                    
                    # Title and info
                    st.markdown(f"**{title[:25]}{'...' if len(title) > 25 else ''}**")
                    st.caption(f"🎭 {genres[:30]}{'...' if len(genres) > 30 else ''}")
                    st.markdown(f"⭐ **{rating:.1f}**/5")
                    
                    # Details button
                    if st.button("🎬 Details & Trailer", key=f"details_{rec_idx}", use_container_width=True):
                        st.session_state.selected_movie_for_details = title

# -------------------------------------------------------------------
# MOVIE DETAILS MODAL
# -------------------------------------------------------------------
if st.session_state.selected_movie_for_details:
    movie_title = st.session_state.selected_movie_for_details
    
    st.markdown("---")
    st.markdown(f"## 🎬 {movie_title}")
    
    # Fetch details from TMDB
    details = get_movie_details(movie_title)
    
    if details:
        col_poster, col_info = st.columns([1, 2])
        
        with col_poster:
            if details.get("poster_path"):
                st.image(f"https://image.tmdb.org/t/p/w500{details['poster_path']}", use_container_width=True)
            else:
                st.markdown(
                    "<div style='height:400px;background:linear-gradient(135deg,#ff6b6b,#4ecdc4);"
                    "border-radius:12px;display:flex;align-items:center;justify-content:center;"
                    "font-size:5rem;'>🎬</div>",
                    unsafe_allow_html=True
                )
        
        with col_info:
            # Movie meta info
            meta_items = []
            if details.get("release_date"):
                meta_items.append(f"📅 {details['release_date'][:4]}")
            if details.get("runtime"):
                meta_items.append(f"⏱️ {details['runtime']} min")
            if details.get("vote_average"):
                meta_items.append(f"⭐ {details['vote_average']:.1f}/10 ({details.get('vote_count', 0)} votes)")
            
            if meta_items:
                st.markdown(f"### {' • '.join(meta_items)}")
            
            if details.get("genres"):
                st.markdown(f"**🎭 Genres:** {', '.join(details['genres'])}")
            
            st.markdown("### 📖 Overview")
            st.write(details.get("overview", "No description available."))
            
            if details.get("cast"):
                st.markdown(f"**🎭 Cast:** {', '.join(details['cast'])}")
        
        # Trailer section
        if details.get("trailer_key"):
            st.markdown("### 🎥 Watch Trailer")
            st.video(f"https://www.youtube.com/watch?v={details['trailer_key']}")
        else:
            st.info("🎬 No trailer available for this movie.")
    else:
        st.warning("Could not fetch movie details. Make sure TMDB API key is configured in `.streamlit/secrets.toml`")
    
    # Close button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("✖️ Close Details", key="close_modal", use_container_width=True):
            st.session_state.selected_movie_for_details = None
            st.rerun()

# -------------------------------------------------------------------
# FOOTER + RANDOM QUOTE
# -------------------------------------------------------------------
st.markdown(
    """
<div style='text-align:center;color:#9ca3af;margin-top:2.5rem;padding-top:1.2rem;border-top:1px solid rgba(148,163,184,0.4);'>
    <div style='font-size:1rem;margin-bottom:0.3rem;'>
        <span style='color:#4ecdc4;'>CINEMATIC AI</span> • TF-IDF • SVD • Hybrid ML • MovieLens 1M
    </div>
</div>
""",
    unsafe_allow_html=True,
)

quotes = [
    "“May the Force be with your movie choices.”",
    "“You’re gonna need a bigger watchlist.”",
    "“I’ll be back… with more recommendations.”",
    "“Roads? Where we’re going, we don’t need roads. Just popcorn.”",
]
st.markdown(
    "<div style='text-align:center;color:#6b7280;font-style:italic;margin-top:0.6rem;'>"
    f"{random.choice(quotes)}"
    "</div>",
    unsafe_allow_html=True,
)
