# 🎬 Cinematic AI - Movie Recommendation System

A hybrid Machine Learning-based Movie Recommendation System using the MovieLens 1M dataset. Combines **Content-Based Filtering** (TF-IDF) and **Collaborative Filtering** (SVD) for intelligent movie suggestions.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **Content-Based Recommendations**: Find movies similar to your favorites using TF-IDF on genres and titles
- **Collaborative Filtering**: Personalized suggestions based on user rating patterns using SVD
- **Hybrid Intelligence**: Combine both approaches with adjustable weighting
- **Genre Filtering**: Filter recommendations by specific genres
- **TMDB Integration**: Fetches movie posters from The Movie Database API
- **Modern UI**: Netflix-inspired dark theme with neon aesthetics

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | scikit-learn, scikit-surprise |
| NLP | TF-IDF Vectorization |
| Recommendation | SVD (Singular Value Decomposition) |
| Web App | Streamlit |
| Data | MovieLens 1M Dataset |

## 📁 Project Structure

```
movie-recommendation-system/
├── data/
│   └── ml-1m/              # MovieLens dataset (movies.csv, ratings.csv, users.csv)
├── models/
│   ├── svd_algo.pkl        # Pre-trained SVD model
│   └── tfidf.pkl           # Pre-trained TF-IDF vectorizer
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   └── 02_recommender_light.ipynb
├── src/
│   ├── data_loader.py      # Data loading utilities
│   └── recommender.py      # Recommendation engine
├── streamlit_app/
│   └── app.py              # Streamlit web application
├── .streamlit/
│   └── secrets.toml.example # API key configuration template
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Conda (optional, for environment.yml)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. **Install dependencies**

   Using pip:
   ```bash
   pip install -r requirements.txt
   ```

   Or using Conda:
   ```bash
   conda env create -f environment.yml
   conda activate movie-recsys
   ```

3. **Configure API key (optional, for movie posters)**
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   Edit `.streamlit/secrets.toml` and add your [TMDB API key](https://www.themoviedb.org/settings/api)

4. **Download the dataset**
   
   Download the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/) and extract to `data/ml-1m/`

5. **Run the application**
   ```bash
   streamlit run streamlit_app/app.py
   ```

6. **Open in browser**
   
   Navigate to `http://localhost:8501`

## 🎮 Usage

### Content-Based Filtering
1. Select "CONTENT-BASED AI" model
2. Search and select movies you enjoy (up to 5)
3. Optionally filter by genres
4. Click "Generate Smart Recommendations"

### Collaborative Filtering
1. Select "COLLABORATIVE AI" model
2. Enter a User ID (1-6040 from MovieLens dataset)
3. Optionally filter by genres
4. Click "Generate Smart Recommendations"

### Hybrid Intelligence
1. Select "HYBRID INTELLIGENCE" model
2. Select movies AND enter a User ID
3. Adjust the Content ↔ Collaborative balance slider
4. Optionally filter by genres
5. Click "Generate Smart Recommendations"

## 🧠 How It Works

### Content-Based Filtering
- Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to vectorize movie titles and genres
- Calculates **cosine similarity** between movie vectors
- Recommends movies most similar to user's selected favorites

### Collaborative Filtering
- Implements **SVD (Singular Value Decomposition)** from the Surprise library
- Learns latent factors from user-item rating matrix
- Predicts ratings for unseen movies based on similar user patterns

### Hybrid Approach
- Normalizes scores from both methods using MinMaxScaler
- Combines scores: `hybrid_score = α × content_score + (1-α) × collab_score`
- Adjustable α (alpha) weight via UI slider

## 📊 Dataset

The system uses the **MovieLens 1M** dataset containing:
- **1,000,209** ratings
- **6,040** users
- **3,706** movies
- Rating scale: 1-5 stars

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `TMDB_API_KEY` | API key for movie posters | Optional |

### Streamlit Secrets

Create `.streamlit/secrets.toml`:
```toml
TMDB_API_KEY = "your_api_key_here"
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MovieLens](https://grouplens.org/datasets/movielens/) for the dataset
- [TMDB](https://www.themoviedb.org/) for movie poster API
- [Surprise Library](https://surpriselib.com/) for collaborative filtering
- [Streamlit](https://streamlit.io/) for the web framework

---

<p align="center">
  Made with ❤️ using Python and Machine Learning
</p>
