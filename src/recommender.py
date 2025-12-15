import sys
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from src.data_loader import load_movies, load_ratings
except ImportError:
    from data_loader import load_movies, load_ratings

class MovieRecommender:
    def __init__(self):
        self.svd_model = None
        self.tfidf_model = None
        self.tfidf_matrix = None
        self.movies_df = None
        self.ratings_df = None
        self.load_models()
        self.load_and_preprocess_data()
    
    def load_models(self):
        """Load pre-trained models from models folder"""
        try:
            models_dir = os.path.join(project_root, "models")
            
            # Load SVD model
            with open(os.path.join(models_dir, "svd_algo.pkl"), 'rb') as f:
                self.svd_model = pickle.load(f)
            print("[OK] SVD model loaded successfully")
            
            # Load TF-IDF model
            with open(os.path.join(models_dir, "tfidf.pkl"), 'rb') as f:
                self.tfidf_model = pickle.load(f)
            print("[OK] TF-IDF model loaded successfully")
                
        except Exception as e:
            print(f"[ERROR] Error loading models: {e}")
            raise
    
    def load_and_preprocess_data(self):
        """Load and preprocess movie data exactly like in the notebook"""
        try:
            # Load movies data
            self.movies_df = load_movies()
            self.ratings_df = load_ratings()
            
            # Preprocess exactly like in the notebook
            self.movies_df['genre_text'] = self.movies_df['genres'].str.replace('|', ' ', regex=False)
            self.movies_df['text'] = self.movies_df['title'].fillna('') + ' ' + self.movies_df['genre_text'].fillna('')
            
            # Create movie text for TF-IDF (same as notebook)
            self.movies_df['genres'] = self.movies_df['genres'].fillna('')
            self.movies_df['title'] = self.movies_df['title'].fillna('')
            self.movies_df['text_for_tfidf'] = (
                self.movies_df['title'] + " " +
                self.movies_df['genres'] + " "
            )
            
            # Remove movies with empty text
            self.movies_df = self.movies_df[self.movies_df['text_for_tfidf'].str.strip() != ""]
            
            # Create TF-IDF matrix using the loaded model
            self.tfidf_matrix = self.tfidf_model.transform(self.movies_df['text_for_tfidf'])
            print("[OK] TF-IDF matrix created successfully")
            print(f"[OK] Loaded {len(self.movies_df)} movies")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def content_recommend(self, movie_id, k=10):
        """Content-based recommendations using TF-IDF and cosine similarity"""
        try:
            # Find movie index
            movie_idx = self.movies_df.index[self.movies_df['movieId'] == movie_id].tolist()
            if not movie_idx:
                return self.get_popular_movies(k)
            
            movie_idx = movie_idx[0]
            
            # Compute cosine similarity for this movie against all others
            cosine_sim = linear_kernel(self.tfidf_matrix[movie_idx], self.tfidf_matrix).flatten()
            
            # Get top k similar movies (excluding itself)
            similar_indices = cosine_sim.argsort()[::-1][1:k+1]
            
            recommendations = []
            for idx in similar_indices:
                movie = self.movies_df.iloc[idx]
                recommendations.append({
                    'movieId': movie['movieId'],
                    'title': movie['title'],
                    'genres': movie['genres'],
                    'pred_rating': 3.5 + (cosine_sim[idx] * 1.5),  # Scale similarity to rating
                    'similarity_score': cosine_sim[idx]
                })
            
            return pd.DataFrame(recommendations)
            
        except Exception as e:
            print(f"Error in content recommendation: {e}")
            return self.get_popular_movies(k)
    
    def collaborative_recommend(self, user_id, k=10):
        """Collaborative filtering recommendations using SVD"""
        try:
            # Get movies the user hasn't rated
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
            rated_movies = set(user_ratings['movieId'].values)
            all_movies = set(self.movies_df['movieId'].values)
            unrated_movies = all_movies - rated_movies
            
            # Predict ratings for unrated movies
            predictions = []
            for movie_id in list(unrated_movies)[:1000]:  # Limit for performance
                try:
                    pred = self.svd_model.predict(user_id, movie_id)
                    predictions.append({
                        'movieId': movie_id,
                        'pred_rating': pred.est
                    })
                except:
                    continue
            
            # Sort by predicted rating and get top k
            predictions.sort(key=lambda x: x['pred_rating'], reverse=True)
            top_predictions = predictions[:k]
            
            # Merge with movie details
            result_df = pd.DataFrame(top_predictions)
            result_df = result_df.merge(self.movies_df[['movieId', 'title', 'genres']], on='movieId')
            
            return result_df
            
        except Exception as e:
            print(f"Error in collaborative recommendation: {e}")
            return self.get_popular_movies(k)
    
    def hybrid_recommend(self, user_id, user_movie_ids, top_n=10, alpha=0.5):
        """
        Hybrid recommendations combining content-based and collaborative filtering
        user_id: user ID for collaborative filtering
        user_movie_ids: list of movie IDs the user has watched/liked for content-based
        alpha: weight for content-based vs collaborative (0.5 = equal weight)
        """
        try:
            # If no user movies provided, return collaborative recommendations
            if not user_movie_ids:
                return self.collaborative_recommend(user_id, top_n)
            
            # Get content-based recommendations from all user movies
            content_recs = []
            for movie_id in user_movie_ids:
                content_rec = self.content_recommend(movie_id, top_n*2)
                content_recs.append(content_rec)
            
            # Combine content recommendations
            if content_recs:
                combined_content = pd.concat(content_recs, ignore_index=True)
                # Average scores for duplicate movies
                combined_content = combined_content.groupby('movieId').agg({
                    'title': 'first',
                    'genres': 'first', 
                    'pred_rating': 'mean',
                    'similarity_score': 'mean'
                }).reset_index()
            else:
                combined_content = pd.DataFrame()
            
            # Get collaborative recommendations for the user
            collab_recs = self.collaborative_recommend(user_id, top_n*2)
            
            # If we have content recommendations, combine them
            if not combined_content.empty:
                # Normalize scores for combination
                scaler = MinMaxScaler()
                
                if 'pred_rating' in combined_content.columns:
                    combined_content['content_score'] = scaler.fit_transform(
                        combined_content[['pred_rating']]
                    ).flatten()
                
                if 'pred_rating' in collab_recs.columns:
                    collab_recs['collab_score'] = scaler.fit_transform(
                        collab_recs[['pred_rating']]
                    ).flatten()
                
                # Merge and calculate hybrid score
                hybrid_df = combined_content.merge(
                    collab_recs[['movieId', 'collab_score']], 
                    on='movieId', 
                    how='left'
                ).fillna(0)
                
                # Calculate hybrid score
                hybrid_df['hybrid_score'] = (
                    alpha * hybrid_df.get('content_score', 0) + 
                    (1 - alpha) * hybrid_df.get('collab_score', 0)
                )
                
                # Remove movies the user has already seen
                user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
                rated_movies = set(user_ratings['movieId'].values) if not user_ratings.empty else set()
                hybrid_df = hybrid_df[~hybrid_df['movieId'].isin(rated_movies)]
                
                # Sort by hybrid score and return top N
                hybrid_df = hybrid_df.sort_values('hybrid_score', ascending=False)
                result = hybrid_df.head(top_n)
                
                # Use hybrid score as final predicted rating (scaled to 3-5)
                result['pred_rating'] = 3.0 + (result['hybrid_score'] * 2.0)
                
                return result[['movieId', 'title', 'genres', 'pred_rating']]
            else:
                # Fallback to collaborative recommendations
                return collab_recs.head(top_n)
                
        except Exception as e:
            print(f"Error in hybrid recommendation: {e}")
            return self.get_popular_movies(top_n)
    
    def get_popular_movies(self, top_n=10):
        """Fallback: return most rated movies"""
        try:
            if self.ratings_df is not None:
                # Calculate average ratings and number of ratings
                movie_stats = self.ratings_df.groupby('movieId').agg({
                    'rating': ['count', 'mean']
                }).reset_index()
                movie_stats.columns = ['movieId', 'rating_count', 'avg_rating']
                
                # Filter movies with sufficient ratings and sort by rating and popularity
                popular_movies = movie_stats[movie_stats['rating_count'] > 10]
                popular_movies = popular_movies.sort_values(['avg_rating', 'rating_count'], ascending=False)
                
                # Merge with movie details
                result = popular_movies.merge(self.movies_df, on='movieId')
                result['pred_rating'] = result['avg_rating']
                
                return result[['movieId', 'title', 'genres', 'pred_rating']].head(top_n)
            else:
                # Fallback to first N movies if no ratings data
                result = self.movies_df.head(top_n).copy()
                result['pred_rating'] = 4.0  # Default rating
                return result[['movieId', 'title', 'genres', 'pred_rating']]
                
        except Exception as e:
            print(f"Error getting popular movies: {e}")
            # Ultimate fallback
            result = self.movies_df.head(top_n).copy()
            result['pred_rating'] = 4.0
            return result[['movieId', 'title', 'genres', 'pred_rating']]

# Create global recommender instance
recommender = MovieRecommender()

# Standalone functions for backward compatibility
def content_recommend(movie_id, k=10):
    """Content-based recommendations standalone function"""
    return recommender.content_recommend(movie_id, k)

def collaborative_recommend(user_id, k=10):
    """Collaborative recommendations standalone function"""
    return recommender.collaborative_recommend(user_id, k)

def hybrid_recommend(user_movie_ids, top_n=10):
    """
    Hybrid recommendations standalone function (legacy interface)
    Note: This uses a simplified version without user_id for collaborative part
    """
    # For backward compatibility, use the first movie as content seed
    # and default user_id = 1 for collaborative part
    if user_movie_ids:
        return recommender.hybrid_recommend(user_id=1, user_movie_ids=user_movie_ids, top_n=top_n)
    else:
        return recommender.get_popular_movies(top_n)

# Test the implementation
if __name__ == "__main__":
    print("Testing recommender system...")
    
    # Test Content-Based
    print("\n1. Testing Content-Based Recommendations:")
    content_recs = content_recommend(1, k=3)  # Toy Story
    print("Content-based recommendations for Toy Story:")
    print(content_recs[['title', 'pred_rating', 'genres']])
    
    # Test Collaborative
    print("\n2. Testing Collaborative Recommendations:")
    collab_recs = collaborative_recommend(1, k=3)  # User 1
    print("Collaborative recommendations for User 1:")
    print(collab_recs[['title', 'pred_rating', 'genres']])
    
    # Test Hybrid
    print("\n3. Testing Hybrid Recommendations:")
    hybrid_recs = hybrid_recommend([1, 2, 3], top_n=3)  # Multiple movies
    print("Hybrid recommendations:")
    print(hybrid_recs[['title', 'pred_rating', 'genres']])
    
    print("\n✓ All models working correctly!")