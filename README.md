# Restaurant Recommendation System

A hybrid recommendation system that combines collaborative filtering and content-based filtering to provide personalized restaurant recommendations with explanations.

## Features

- **Hybrid Recommendation Engine**: Combines collaborative filtering (LightFM) with content-based filtering using user and restaurant features
- **Explainable Recommendations**: Provides human-readable explanations for why restaurants are recommended
- **Multiple Evaluation Strategies**: 
  - Random split evaluation
  - Cold start user evaluation
  - Leave-one-out cross-validation
- **Interactive Demo**: Command-line interface to test recommendations for different users
- **Comprehensive Metrics**: Precision@K, Recall@K, AUC, and diversity scoring
- **Content Matching**: Evaluates how well recommendations match user preferences for new users

##  System Architecture

The system uses a hybrid approach combining:

1. **Collaborative Filtering**: Uses LightFM with WARP loss to learn user-item interactions
2. **Content-Based Features**: 
   - User features: smoking preference, dress preference, budget, activity, cuisine preferences
   - Restaurant features: cuisine types, smoking area, dress code, price range, location
3. **Feature Engineering**: One-hot encoding for categorical features, multi-label encoding for cuisines
4. **Evaluation Framework**: Multiple splits to test different scenarios including cold start problems

##  Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/restaurant-recommendation-system.git
   cd restaurant-recommendation-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Setup**
   
   Download the restaurant dataset and place the following CSV files in a `dataset/` folder:
   - `chefmozcuisine.csv` - Restaurant cuisine information
   - `userprofile.csv` - User profile data
   - `geoplaces2.csv` - Restaurant profile data
   - `rating_final.csv` - User-restaurant ratings
   - `usercuisine.csv` - User cuisine preferences

   **Dataset Source**: [Restaurant & Consumer Data](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data)

##  Quick Start

### Basic Usage

```python
from src.recommendation_system import *

# Load and preprocess data
restaurant_cuisine = pd.read_csv('dataset/chefmozcuisine.csv')
user_profiles = pd.read_csv('dataset/userprofile.csv')
restaurant_profiles = pd.read_csv('dataset/geoplaces2.csv')
ratings = pd.read_csv('dataset/rating_final.csv')

# Train the model and run evaluation
results = train_and_evaluate_hybrid_recommender(ratings, user_profiles, restaurant_profiles)

# Create production model
model, user_id_map, restaurant_id_map, user_features, restaurant_features = define_model_parameters(
    ratings, user_profiles, restaurant_profiles
)

# Get recommendations for a user
recommendations = explain_restaurant_recommendations(
    user_id=1001, 
    model=model,
    user_id_map=user_id_map,
    restaurant_id_map=restaurant_id_map,
    restaurant_profiles_df=restaurant_profiles,
    user_profiles_df=user_profiles,
    user_features=user_features,
    restaurant_features=restaurant_features,
    ratings_df=ratings,
    n_recommendations=5
)
```

### Interactive Demo

Run the interactive demo to test recommendations:

```python
# Launch interactive demo
interactive_recommendation_demo(
    model=model,
    user_id_map=user_id_map,
    restaurant_id_map=restaurant_id_map,
    restaurant_profiles_df=restaurant_profiles,
    user_profiles_df=user_profiles,
    ratings_df=ratings,
    user_features=user_features,
    restaurant_features=restaurant_features
)
```

##  Model Performance

The system is evaluated using multiple strategies:

| Metric | Random Split | Cold Start | Leave-One-Out |
|--------|-------------|------------|---------------|
| Precision@10 | ~0.85 | ~0.72 | ~0.88 |
| Recall@10 | ~0.45 | ~0.38 | ~0.47 |
| AUC | ~0.92 | ~0.89 | ~0.93 |
| Diversity | ~0.76 | ~0.78 | ~0.75 |

*Note: Actual performance may vary based on dataset and hyperparameters*

##  Configuration

### Model Hyperparameters

```python
model = LightFM(
    no_components=350,        # Number of latent factors
    loss='warp',             # WARP loss for ranking
    learning_schedule='adagrad',  # Adaptive learning rate
    learning_rate=0.05       # Initial learning rate
)
```

### Key Parameters
- `no_components`: Controls model complexity (default: 350)
- `epochs`: Training iterations (default: 100)
- `learning_rate`: Controls learning speed (default: 0.05)

##  Evaluation Metrics

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended  
- **AUC**: Area Under the ROC Curve
- **Diversity**: Measures how different recommended items are from each other
- **Content Match**: How well recommendations match user preferences (cold start evaluation)

##  Evaluation Strategies

1. **Random Split**: Standard 80/20 train-test split
2. **Cold Start**: Evaluates performance on users not seen during training
3. **Leave-One-Out**: Holds out one rating per user for testing


## References

- [LightFM Documentation](https://making.lyst.com/lightfm/docs/home.html)
- [Restaurant & Consumer Data](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data)

