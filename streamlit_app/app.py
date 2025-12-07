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
TMDB_API_KEY = "733a194c5a85ab47618704e56b14ebff"

# Allow Streamlit secrets or env variables to override only if they exist.
try:
    TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", TMDB_API_KEY)
except Exception:
    TMDB_API_KEY = os.getenv("TMDB_API_KEY", TMDB_API_KEY)

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
# GLOBAL CSS  (full, with glow + Netflix row)
# -------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Exo+2:wght@300;400;600&display=swap');

body {
    background: radial-gradient(circle at top, #1a1a2e 0, #0b0c10 45%, #000 100%);
    color: #ffffff;
    font-family: 'Exo 2', sans-serif;
}

.app-container {
    max-width: 1200px;
    margin: 0 auto;
}

/* MAIN HEADER */
.neon-header {
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
}

.neon-subtitle {
    color: #b0b0b0;
    font-size: 1.1rem;
}

@keyframes hdr-pulse {
    0% { text-shadow: 0 0 6px rgba(78,205,196,0.5); }
    100% { text-shadow: 0 0 24px rgba(78,205,196,0.9); }
}

/* SECTION HEADER */
.section-header {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.7rem;
    color: #4ecdc4;
    text-align: center;
    margin: 1.5rem 0 1rem 0;
    text-shadow: 0 0 8px rgba(78,205,196,0.8);
}

/* MODEL CARDS */
.model-card {
    background: rgba(16, 24, 40, 0.9);
    border-radius: 18px;
    border: 1px solid rgba(148,163,184,0.4);
    padding: 1.2rem 1.3rem;
    margin-bottom: 1.1rem;
    backdrop-filter: blur(14px);
    box-shadow:
        0 0 0 1px rgba(15,23,42,0.7),
        0 12px 35px rgba(15,23,42,0.95);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    transform-style: preserve-3d;
    transform-origin: center;
}

.model-card:hover {
    transform: translateY(-6px) rotate3d(1, -1, 0, 6deg);
    box-shadow:
        0 0 0 1px rgba(94,234,212,0.4),
        0 18px 55px rgba(56,189,248,0.55);
    border-color: #4ecdc4;
    cursor: pointer;
}

.model-card.selected {
    border-color: #ff6b6b;
    box-shadow:
        0 0 0 1px rgba(248,113,113,0.6),
        0 18px 60px rgba(248,113,113,0.85);
    animation: neon-pulse 2.2s ease-in-out infinite alternate;
}

@keyframes neon-pulse {
    0% {
        box-shadow:
            0 0 0 1px rgba(248,113,113,0.4),
            0 14px 40px rgba(248,113,113,0.5);
    }
    100% {
        box-shadow:
            0 0 0 2px rgba(248,113,113,0.7),
            0 18px 64px rgba(248,113,113,0.9);
    }
}

.model-icon {
    font-size: 2.4rem;
    margin-bottom: 0.4rem;
}

.model-title {
    font-family: 'Orbitron';
    font-size: 1.3rem;
    font-weight: 700;
    color: #e5e7eb;
    margin-bottom: 0.2rem;
}

.model-desc {
    font-size: 0.9rem;
    color: #9ca3af;
}

/* HORIZONTAL RECOMMENDATION ROW */
.rec-row-wrapper {
    position: relative;
    margin-top: 1rem;
}

.rec-row {
    display: flex;
    overflow-x: auto;
    gap: 1rem;
    padding-bottom: 0.5rem;
    scroll-behavior: smooth;
}

.rec-row::-webkit-scrollbar {
    height: 8px;
}
.rec-row::-webkit-scrollbar-track {
    background: rgba(15,23,42,0.9);
}
.rec-row::-webkit-scrollbar-thumb {
    background: linear-gradient(90deg, #4ecdc4, #ff6b6b);
    border-radius: 999px;
}

.rec-card {
    flex: 0 0 300px;
    max-width: 220px;
    height: 500px;  
    background: radial-gradient(circle at top left, rgba(56,189,248,0.16), rgba(15,23,42,0.96));
    border-radius: 16px;
    border: 1px solid rgba(148,163,184,0.4);
    overflow: hidden;
    box-shadow: 0 14px 40px rgba(15,23,42,0.9);
    backdrop-filter: blur(10px);
    display: flex;
    flex-direction: column;
}

.rec-poster {
    width: 100%;
    height: 800px;
    background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
    overflow: hidden;
}

.rec-poster img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}

.rec-body {
    padding: 0.7rem 0.75rem 0.8rem 0.75rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
}

.movie-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e5e7eb;
    margin-bottom: 0.2rem;
}

.movie-genres {
    font-size: 0.78rem;
    color: #67e8f9;
    margin-bottom: 0.3rem;
}

.movie-rating {
    font-family: 'Orbitron';
    font-size: 0.85rem;
    color: #facc15;
}

.movie-meta-sub {
    font-size: 0.7rem;
    color: #9ca3af;
    text-align: right;
}

.pill {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 0.1rem 0.5rem;
    font-size: 0.7rem;
    background: rgba(15,23,42,0.9);
    color: #e5e7eb;
    border: 1px solid rgba(148,163,184,0.6);
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------
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
                st.markdown(
                    f"<h2 style='color:#4ecdc4;'>{title_prefix}</h2>",
                    unsafe_allow_html=True,
                )

                # Build the entire horizontal row HTML as a single flat string
                row_html = '<div class="rec-row-wrapper"><div class="rec-row">'
                for idx, row in recs.iterrows():
                    title = str(row["title"])
                    genres = str(row["genres"])
                    rating = float(row.get("pred_rating", 4.0))
                    poster_url = get_poster_url(title)

                    if poster_url:
                        img_html = "<img src='" + poster_url + "' alt='Poster' />"
                    else:
                        img_html = ""

                    card_html = (
                        f"<div class='rec-card' style='animation-delay:{idx*0.08}s'>"
                        "<div class='rec-poster'>"
                        f"{img_html}"
                        "</div>"
                        "<div class='rec-body'>"
                        "<div>"
                        f"<div class='movie-title'>{title}</div>"
                        f"<div class='movie-genres'>{genres}</div>"
                        "</div>"
                        "<div style='display:flex;justify-content:space-between;align-items:center;margin-top:0.25rem;'>"
                        f"<div class='movie-rating'>⭐ {rating:.2f}/5</div>"
                        "<div class='movie-meta-sub'><span class='pill'>AI match</span></div>"
                        "</div>"
                        "</div>"
                        "</div>"
                    )
                    row_html += card_html

                row_html += "</div></div>"

                st.markdown(row_html, unsafe_allow_html=True)
            else:
                st.info("No recommendations to display yet. Try adjusting your inputs.")
        except Exception as e:
            st.error(f"❌ Error generating recommendations: {e}")

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
