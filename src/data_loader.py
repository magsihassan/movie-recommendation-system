import pandas as pd
import os

DATA_PATH = os.path.join("data", "ml-1m")

def load_movies():
    movies_path = os.path.join(DATA_PATH, "movies.csv")
    try:
        # First, let's inspect the file to understand its structure
        with open(movies_path, 'r', encoding='latin-1') as f:
            first_lines = [f.readline() for _ in range(5)]
            print("First 5 lines of the file:")
            for i, line in enumerate(first_lines):
                print(f"Line {i+1}: {line.strip()}")
        
        # Try different delimiters and formats
        # MovieLens 1M typically uses :: as delimiter
        try:
            print("Trying :: delimiter...")
            movies = pd.read_csv(movies_path, sep='::', engine='python', 
                               names=["movieId", "title", "genres"], encoding='latin-1')
            print("Success with :: delimiter")
            return movies
        except Exception as e:
            print(f":: delimiter failed: {e}")
            
        # Try comma delimiter
        try:
            print("Trying comma delimiter...")
            movies = pd.read_csv(movies_path, sep=',', 
                               names=["movieId", "title", "genres"], encoding='latin-1')
            print("Success with comma delimiter")
            return movies
        except Exception as e:
            print(f"Comma delimiter failed: {e}")
            
        # Try tab delimiter
        try:
            print("Trying tab delimiter...")
            movies = pd.read_csv(movies_path, sep='\t', 
                               names=["movieId", "title", "genres"], encoding='latin-1')
            print("Success with tab delimiter")
            return movies
        except Exception as e:
            print(f"Tab delimiter failed: {e}")
            
        # Last resort: read with error handling
        print("Trying with error handling...")
        movies = pd.read_csv(movies_path, encoding='latin-1', error_bad_lines=False, warn_bad_lines=True)
        return movies
        
    except Exception as e:
        print(f"Error loading movies: {e}")
        raise

def load_ratings():
    ratings_path = os.path.join(DATA_PATH, "ratings.csv")
    try:
        # Try different delimiters for ratings too
        try:
            ratings = pd.read_csv(ratings_path, sep='::', engine='python',
                                names=["userId", "movieId", "rating", "timestamp"], encoding='latin-1')
            return ratings
        except:
            ratings = pd.read_csv(ratings_path, sep=',',
                                names=["userId", "movieId", "rating", "timestamp"], encoding='latin-1')
            return ratings
    except Exception as e:
        print(f"Error loading ratings: {e}")
        raise