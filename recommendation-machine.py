# Import necessary libraries
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from lightfm import LightFM
from scipy import sparse
from lightfm.evaluation import precision_at_k, recall_at_k
from lightfm.evaluation import auc_score



# Load all required datasets
restaurant_cuisine = pd.read_csv('dataset/chefmozcuisine.csv')
user_profiles = pd.read_csv('dataset/userprofile.csv')
restaurant_profiles = pd.read_csv('dataset/geoplaces2.csv')
ratings = pd.read_csv('dataset/rating_final.csv')
user_cuisine = pd.read_csv('dataset/usercuisine.csv')

# Display basic information about user profiles
print(f"User profiles loaded: {user_profiles.count().iloc[0]} records")

# Process user cuisine preferences
# Load user cuisine data and group by userID to handle multiple cuisines per user

# Group cuisines by user and join them with '-' separator
user_cuisine = user_cuisine.groupby('userID')['Rcuisine'].apply(lambda x: '-'.join(x.unique())).reset_index()
print(f"User cuisine preferences processed: {user_cuisine.count().iloc[0]} records")

# Merge user profiles with cuisine preferences
user_profiles = user_profiles.merge(user_cuisine, on='userID', how='left')
# Fill missing cuisine preferences with 'unknown'
user_profiles['Rcuisine'] = user_profiles['Rcuisine'].fillna('unknown')
print("User profiles with cuisine preferences:")
print(user_profiles.head(20))


# Process restaurant cuisine information
# Group cuisines by restaurant and join them with '-' separator
restaurant_cuisine_grouped = restaurant_cuisine.groupby('placeID')['Rcuisine'].apply(lambda x: '-'.join(x.unique())).reset_index()
print("Restaurant cuisine data grouped:")
print(restaurant_cuisine_grouped.head(20))

# Merge restaurant profiles with cuisine information
restaurant_profiles = restaurant_profiles.merge(restaurant_cuisine_grouped, on='placeID', how='left')
# Save a sample of restaurant data for inspection
restaurant_head = restaurant_profiles.head()
print("Restaurant profiles with cuisine information:")
print(restaurant_head)

# Display processed restaurant and user data
print("Final restaurant profiles structure:")
print(restaurant_profiles.head())

print("Final user profiles structure:")
print(user_profiles.head())

# ===== 2. FEATURE ENGINEERING =====

def create_user_features(user_profiles_df):
    """
    Create a comprehensive feature matrix for users based on their profiles.
    
    This function combines:
    1. Identity features (one-hot encoding for each user)
    2. Content features (categorical attributes like preferences)
    
    Args:
        user_profiles_df (DataFrame): User profile data
    
    Returns:
        tuple: (user_features_matrix, user_id_mapping)
    """
    from sklearn.preprocessing import OneHotEncoder
    from scipy import sparse

    # 1. Create identity features - each user gets a unique one-hot vector
    # This allows the model to learn user-specific patterns
    identity_features = sparse.identity(len(user_profiles_df))

    # 2. Preprocess text data to handle case sensitivity
    user_data = user_profiles_df.copy()
    if 'Rcuisine' in user_data.columns:
        # Convert cuisine preferences to lowercase for consistency
        user_data['Rcuisine'] = user_data['Rcuisine'].astype(str).str.lower()

    # 3. Define categorical features to encode
    categorical_features = ['smoker', 'dress_preference', 'budget', 'activity', 'Rcuisine']
    categorical_data = user_data[categorical_features].fillna('unknown')

    # Normalize text fields to lowercase for consistency
    categorical_data['dress_preference'] = categorical_data['dress_preference'].astype(str).str.lower()
    categorical_data['activity'] = categorical_data['activity'].astype(str).str.lower()

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    content_features = encoder.fit_transform(categorical_data)

    # 4. Combine identity and content features
    # This gives the model both user-specific and content-based information
    user_features = sparse.hstack([identity_features, content_features])

    # Create mapping from user ID to matrix row index
    user_id_map = {user_id: i for i, user_id in enumerate(user_profiles_df['userID'])}

    return user_features.tocsr(), user_id_map

def create_restaurant_features(restaurant_profiles_df):
    """
    Create a comprehensive feature matrix for restaurants based on their profiles.
    
    This function combines:
    1. Identity features (unique encoding for each restaurant)
    2. Cuisine features (multi-label encoding for multiple cuisines)
    3. Other categorical features (location, price, etc.)
    
    Args:
        restaurant_profiles_df (DataFrame): Restaurant profile data
    
    Returns:
        tuple: (restaurant_features_matrix, restaurant_id_mapping)
    """
    from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
    from scipy import sparse
    import pandas as pd

    # 1. Create identity features - each restaurant gets a unique identifier
    # This allows the model to learn restaurant-specific patterns
    identity_features = sparse.identity(len(restaurant_profiles_df))

    # 2. Process cuisine information
    # Handle multiple cuisines per restaurant (e.g., "Italian-Mexican-Vietnamese")
    restaurant_profiles_df = restaurant_profiles_df.copy()
    restaurant_profiles_df['Rcuisine'] = restaurant_profiles_df['Rcuisine'].astype(str).str.lower()
    
    # Split cuisine strings and create multi-label binary encoding
    cuisine_lists = [c.split('-') for c in restaurant_profiles_df['Rcuisine'].fillna('unknown')]
    cuisine_features = sparse.csr_matrix(MultiLabelBinarizer().fit_transform(cuisine_lists))

    # 3. Process other categorical features
    categorical_features = ['smoking_area', 'dress_code', 'city', 'country', 'state', 'price']
    categorical_data = restaurant_profiles_df[categorical_features].fillna('unknown')

    # Normalize text fields to lowercase for consistency
    categorical_data['city'] = categorical_data['city'].astype(str).str.lower()
    categorical_data['country'] = categorical_data['country'].astype(str).str.lower()
    categorical_data['state'] = categorical_data['state'].astype(str).str.lower()

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    other_features = encoder.fit_transform(categorical_data)

    # 4. Combine all feature types: Identity + Cuisine + Other attributes
    restaurant_features = sparse.hstack([identity_features, cuisine_features, other_features])

    # Create mapping from restaurant ID to matrix row index
    restaurant_id_map = {place_id: i for i, place_id in enumerate(restaurant_profiles_df['placeID'])}

    return restaurant_features.tocsr(), restaurant_id_map

# Generate feature matrices for users and restaurants
print("Creating user and restaurant feature matrices...")
user_features, user_id_map = create_user_features(user_profiles)
restaurant_features, restaurant_id_map = create_restaurant_features(restaurant_profiles)

# Display feature matrices in array format for understanding
print("User features matrix (first few rows):")
print(user_features.toarray())
print("\nRestaurant features matrix (first few rows):")
print(restaurant_features.toarray())
# Note: Each row represents a user/restaurant, each column represents a one-hot encoded feature
# Data is stored in sparse matrix format for memory efficiency

# ===== 3. INTERACTION MATRIX CREATION =====

def create_interaction_matrix(ratings_df, user_id_map, restaurant_id_map, use_weighted=True):
    """
    Create a sparse user-restaurant interaction matrix from ratings data.
    
    This matrix represents user-restaurant interactions where:
    - Rows represent users
    - Columns represent restaurants  
    - Values represent rating scores
    
    Args:
        ratings_df (DataFrame): User-restaurant ratings data
        user_id_map (dict): Mapping from user IDs to matrix row indices
        restaurant_id_map (dict): Mapping from restaurant IDs to matrix column indices
        use_weighted (bool): Whether to use weighted ratings (currently not implemented)
    
    Returns:
        sparse.csr_matrix: User-restaurant interaction matrix
    """
    # Get matrix dimensions
    n_users = len(user_id_map)
    n_items = len(restaurant_id_map)

    # Filter ratings to only include users and restaurants in our feature matrices
    # This ensures consistency between interaction matrix and feature matrices
    valid_ratings = ratings_df[
        ratings_df['userID'].isin(user_id_map.keys()) &
        ratings_df['placeID'].isin(restaurant_id_map.keys())
    ]
    
    # Report data filtering statistics
    print(f"Valid ratings: {len(valid_ratings)}/{len(ratings_df)} ({len(valid_ratings)/len(ratings_df)*100:.1f}%)")
    if valid_ratings.empty:
        print("Warning: No valid ratings found after filtering!")
        return sparse.csr_matrix((n_users, n_items))

    # Map user and restaurant IDs to matrix indices
    user_indices = [user_id_map[user_id] for user_id in valid_ratings['userID']]
    item_indices = [restaurant_id_map[place_id] for place_id in valid_ratings['placeID']]
    ratings = valid_ratings['rating'].values

    # Create sparse interaction matrix
    # COO (Coordinate) format is efficient for construction
    # CSR (Compressed Sparse Row) format is efficient for computation
    interactions = sparse.coo_matrix(
        (ratings, (user_indices, item_indices)),
        shape=(n_users, n_items)
    )

    return interactions.tocsr()

# Create the main interaction matrix
print("Creating user-restaurant interaction matrix...")
interaction_matrix = create_interaction_matrix(ratings, user_id_map, restaurant_id_map)

# ===== 4. EVALUATION FRAMEWORK =====

def create_multiple_evaluation_splits(ratings_df, user_profiles_df, restaurant_profiles_df):
    """
    Create multiple evaluation splits to comprehensively assess the recommender system.
    
    Different splits test different aspects:
    1. Random split - General performance
    2. Cold start split - Performance on new users
    3. Leave-one-out - Robustness across individual users
    
    Args:
        ratings_df (DataFrame): User-restaurant ratings
        user_profiles_df (DataFrame): User profile data
        restaurant_profiles_df (DataFrame): Restaurant profile data
    
    Returns:
        dict: Dictionary containing different evaluation splits
    """
    evaluation_splits = {}

    # 1. Random Split - Standard train/test split
    # Tests general model performance on randomly selected data
    train_random, test_random = train_test_split(ratings_df, test_size=0.2, random_state=42)
    evaluation_splits['random'] = (train_random, test_random)

    # 2. Cold Start Split - Test performance on new users
    # Simulates real-world scenario where new users join the system
    unique_users = ratings_df['userID'].unique()
    np.random.seed(42)
    # Select 20% of users as "new users" who weren't in training
    cold_start_users = np.random.choice(unique_users, size=int(len(unique_users) * 0.2), replace=False)

    # Training set excludes cold start users
    train_user = ratings_df[~ratings_df['userID'].isin(cold_start_users)].copy()
    # Test set contains only cold start users
    test_user = ratings_df[ratings_df['userID'].isin(cold_start_users)].copy()

    evaluation_splits['cold_start'] = (train_user, test_user, cold_start_users)

    # 3. Leave-One-Out Cross-Validation
    # For each user, hold out one rating for testing
    # Tests model's ability to predict individual user preferences
    train_loo = []
    test_loo = []

    for user_id in unique_users:
        user_ratings = ratings_df[ratings_df['userID'] == user_id]

        # Skip users with only one rating (can't leave one out)
        if len(user_ratings) <= 1:
            train_loo.append(user_ratings)
            continue

        # Randomly select one rating for testing
        user_ratings = user_ratings.sample(frac=1, random_state=42)  # Shuffle ratings
        test_item = user_ratings.iloc[[0]]  # First item for testing
        train_items = user_ratings.iloc[1:]  # Rest for training

        test_loo.append(test_item)
        train_loo.append(train_items)

    # Combine all user data back into single dataframes
    train_loo_df = pd.concat(train_loo, ignore_index=True)
    test_loo_df = pd.concat(test_loo, ignore_index=True)

    evaluation_splits['leave_one_out'] = (train_loo_df, test_loo_df)

    return evaluation_splits

def evaluate_recommender(model, train_interactions, test_interactions,
                         user_features, restaurant_features,
                         user_id_map, restaurant_id_map,
                         k=10):
    """
    Evaluate recommender model using multiple metrics.
    
    Metrics include:
    - Precision@K: Fraction of recommended items that are relevant
    - Recall@K: Fraction of relevant items that are recommended
    - AUC: Area Under the ROC Curve
    - Diversity: How different the recommended items are from each other
    
    Args:
        model: Trained LightFM model
        train_interactions: Training interaction matrix
        test_interactions: Test interaction matrix
        user_features: User feature matrix
        restaurant_features: Restaurant feature matrix
        user_id_map: User ID to index mapping
        restaurant_id_map: Restaurant ID to index mapping
        k (int): Number of recommendations to evaluate
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Calculate precision at k - what fraction of recommendations are relevant
    precision = precision_at_k(model, test_interactions, k=k,
                              user_features=user_features,
                              item_features=restaurant_features).mean()

    # Calculate recall at k - what fraction of relevant items are recommended
    recall = recall_at_k(model, test_interactions, k=k,
                        user_features=user_features,
                        item_features=restaurant_features).mean()

    # Calculate AUC - ability to distinguish between relevant and irrelevant items
    auc = auc_score(model, test_interactions,
                   user_features=user_features,
                   item_features=restaurant_features).mean()

    # Calculate diversity - how different recommended items are from each other
    diversity_score = calculate_diversity(model, user_features, restaurant_features, k)

    return {
        'precision@k': precision,
        'recall@k': recall,
        'auc': auc,
        'diversity': diversity_score
    }

def calculate_diversity(model, user_features, restaurant_features, k=10):
    """
    Calculate recommendation diversity by measuring feature similarity between recommended items.
    
    Higher diversity means recommended items are more different from each other.
    This prevents the system from recommending only very similar items.
    
    Args:
        model: Trained LightFM model
        user_features: User feature matrix
        restaurant_features: Restaurant feature matrix
        k (int): Number of recommendations to analyze
    
    Returns:
        float: Average diversity score (0-1, higher is more diverse)
    """
    n_users = user_features.shape[0]
    n_items = restaurant_features.shape[0]
    
    # Sample a subset of users for efficiency
    user_indices = np.random.choice(n_users, size=min(100, n_users), replace=False)
    diversity_scores = []

    for user_idx in user_indices:
        # Create arrays for prediction - one user against all items
        user_ids_array = np.full(n_items, user_idx)
        
        # Get prediction scores for all restaurants for this user
        scores = model.predict(
            user_ids=user_ids_array,
            item_ids=np.arange(n_items),
            user_features=user_features,
            item_features=restaurant_features
        )
        
        # Get top-k recommendations
        top_items = np.argsort(-scores)[:k]

        if len(top_items) <= 1:
            continue

        # Extract feature vectors for recommended items
        item_features = restaurant_features[top_items].toarray()

        # Calculate pairwise cosine similarity between recommended items
        similarity_sum = 0
        count = 0

        for i in range(len(item_features)):
            for j in range(i+1, len(item_features)):
                sim = cosine_similarity([item_features[i]], [item_features[j]])[0][0]
                similarity_sum += sim
                count += 1

        # Diversity is inverse of similarity (higher diversity = lower similarity)
        avg_similarity = similarity_sum / count if count > 0 else 0
        diversity = 1 - avg_similarity
        diversity_scores.append(diversity)

    return np.mean(diversity_scores) if diversity_scores else 0.0

def evaluate_content_match(model, cold_start_users, user_profiles_df, restaurant_profiles_df,
                          user_features, restaurant_features, user_id_map, restaurant_id_map):
    """
    Evaluate how well content-based recommendations match user preferences for cold start users.
    
    This specifically tests the system's ability to make good recommendations for new users
    based solely on their profile information (no historical ratings).
    
    Args:
        model: Trained LightFM model
        cold_start_users: List of user IDs considered as new users
        user_profiles_df: User profile data
        restaurant_profiles_df: Restaurant profile data
        user_features: User feature matrix
        restaurant_features: Restaurant feature matrix
        user_id_map: User ID to index mapping
        restaurant_id_map: Restaurant ID to index mapping
    
    Returns:
        dict: Content match evaluation results
    """
    # Get profiles for cold start users
    cold_user_profiles = user_profiles_df[user_profiles_df['userID'].isin(cold_start_users)]

    # Filter to users that exist in our mapping
    valid_users = [user for user in cold_start_users if user in user_id_map]

    if not valid_users:
        return {"error": "No valid cold start users found"}

    # Evaluate content matching for each cold start user
    n_items = restaurant_features.shape[0]
    content_match_scores = []

    for user_id in valid_users:
        user_idx = user_id_map[user_id]
        user_profile = cold_user_profiles[cold_user_profiles['userID'] == user_id].iloc[0]

        # Get top-10 recommendations for this user
        scores = model.predict(
            user_idx,
            np.arange(n_items),
            user_features=user_features,
            item_features=restaurant_features
        )
        top_items = np.argsort(-scores)[:10]

        # Extract user preferences
        smoker_pref = user_profile['smoker'] if 'smoker' in user_profile else None
        dress_pref = user_profile['dress_preference'] if 'dress_preference' in user_profile else None
        budget_pref = user_profile['budget'] if 'budget' in user_profile else None

        # Get restaurant profiles for recommendations
        rev_map = {idx: place_id for place_id, idx in restaurant_id_map.items()}
        rec_place_ids = [rev_map[idx] for idx in top_items if idx in rev_map]
        rec_profiles = restaurant_profiles_df[restaurant_profiles_df['placeID'].isin(rec_place_ids)]

        # Calculate content match score
        match_score = 0

        # Check smoking preference match
        if smoker_pref is not None and 'smoking_area' in rec_profiles.columns:
            smoking_matches = 0
            for _, rest in rec_profiles.iterrows():
                # If user smokes and restaurant allows smoking
                if (smoker_pref == 'TRUE' and
                    rest['smoking_area'] not in ['none', 'not permitted']):
                    smoking_matches += 1
                # If user doesn't smoke and restaurant is non-smoking
                elif (smoker_pref == 'FALSE' and
                      rest['smoking_area'] in ['none', 'not permitted']):
                    smoking_matches += 1

            match_score += smoking_matches / len(rec_profiles) if len(rec_profiles) > 0 else 0

        # Check dress preference match
        if dress_pref is not None and 'dress_code' in rec_profiles.columns:
            dress_matches = 0
            for _, rest in rec_profiles.iterrows():
                if dress_pref == rest['dress_code']:
                    dress_matches += 1

            match_score += dress_matches / len(rec_profiles) if len(rec_profiles) > 0 else 0

        # Check budget preference match
        if budget_pref is not None and 'price' in rec_profiles.columns:
            budget_matches = 0
            # Define budget-price mapping
            budget_map = {
                'low': ['low'],
                'medium': ['medium-low', 'medium'],
                'high': ['medium-high', 'high']
            }

            for _, rest in rec_profiles.iterrows():
                if rest['price'] in budget_map.get(budget_pref, []):
                    budget_matches += 1

            match_score += budget_matches / len(rec_profiles) if len(rec_profiles) > 0 else 0

        # Normalize match score by number of features checked
        num_features = sum([smoker_pref is not None, dress_pref is not None, budget_pref is not None])
        normalized_score = match_score / num_features if num_features > 0 else 0
        content_match_scores.append(normalized_score)

    return {
        'avg_content_match': np.mean(content_match_scores) if content_match_scores else 0,
        'num_users_evaluated': len(content_match_scores),
        'individual_scores': content_match_scores
    }

# ===== 5. MAIN TRAINING AND EVALUATION FUNCTION =====

def train_and_evaluate_hybrid_recommender(ratings_df, user_profiles_df, restaurant_profiles_df):
    """
    Train and evaluate the hybrid recommender system using multiple evaluation strategies.
    
    This function:
    1. Creates user and restaurant feature matrices
    2. Generates different evaluation splits
    3. Trains LightFM models for each split
    4. Evaluates performance using multiple metrics
    5. Provides comprehensive performance analysis
    
    Args:
        ratings_df: User-restaurant ratings data
        user_profiles_df: User profile data
        restaurant_profiles_df: Restaurant profile data
    
    Returns:
        dict: Comprehensive evaluation results for different split strategies
    """
    # Create feature matrices
    print("Creating feature matrices...")
    user_features, user_id_map = create_user_features(user_profiles_df)
    restaurant_features, restaurant_id_map = create_restaurant_features(restaurant_profiles_df)

    # Create different evaluation splits
    print("Creating evaluation splits...")
    evaluation_splits = create_multiple_evaluation_splits(
        ratings_df, user_profiles_df, restaurant_profiles_df
    )

    results = {}

    # Train and evaluate model for each split strategy
    for split_name, split_data in evaluation_splits.items():
        print(f"\nEvaluating with {split_name} split strategy...")

        # Handle different split data structures
        if split_name == 'cold_start':
            train_ratings, test_ratings, cold_start_users = split_data
        else:
            train_ratings, test_ratings = split_data

        # Create interaction matrices for training and testing
        train_interactions = create_interaction_matrix(
            train_ratings, user_id_map, restaurant_id_map, use_weighted=True
        )
        test_interactions = create_interaction_matrix(
            test_ratings, user_id_map, restaurant_id_map, use_weighted=True
        )

        # Train LightFM model with optimized hyperparameters
        print(f"Training model for {split_name} split...")
        model = LightFM(
            no_components=350,  # Number of latent factors
            loss='warp',        # WARP loss for ranking
            learning_schedule='adagrad',  # Adaptive learning rate
            learning_rate=0.05  # Initial learning rate
        )

        # Fit model with both interaction data and feature matrices
        model.fit(
            interactions=train_interactions,
            user_features=user_features,
            item_features=restaurant_features,
            epochs=100,  # Number of training epochs
            verbose=True
        )

        # Evaluate model performance
        print(f"Evaluating model for {split_name} split...")
        metrics = evaluate_recommender(
            model, train_interactions, test_interactions,
            user_features, restaurant_features,
            user_id_map, restaurant_id_map
        )

        # Additional evaluation for cold start scenario
        if split_name == 'cold_start':
            print("Evaluating content matching for cold start users...")
            content_eval = evaluate_content_match(
                model, cold_start_users, user_profiles_df, restaurant_profiles_df,
                user_features, restaurant_features, user_id_map, restaurant_id_map
            )
            metrics['content_match'] = content_eval['avg_content_match']

        # Store results
        results[split_name] = {
            'model': model,
            'metrics': metrics
        }

        # Print performance metrics
        print(f"Results for {split_name} split:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

    return results

# ===== 6. RUN MAIN EVALUATION =====
print("Starting comprehensive evaluation of hybrid recommender system...")
results = train_and_evaluate_hybrid_recommender(ratings, user_profiles, restaurant_profiles)

# ===== 7. RESULTS COMPARISON =====
print("\n=== COMPREHENSIVE RESULTS SUMMARY ===")
metrics_comparison = {}

# Organize results by metric for easy comparison
for split_name, split_results in results.items():
    for metric_name, metric_value in split_results['metrics'].items():
        if metric_name not in metrics_comparison:
            metrics_comparison[metric_name] = {}
        metrics_comparison[metric_name][split_name] = metric_value

# Display comparison across all evaluation strategies
print("Performance comparison across different evaluation strategies:")
for metric_name, split_values in metrics_comparison.items():
    print(f"\n{metric_name.upper()}:")
    for split_name, value in split_values.items():
        print(f"  {split_name}: {value:.4f}")

# ===== 8. PRODUCTION MODEL CREATION =====

def define_model_parameters(ratings_df, user_profiles_df, restaurant_profiles_df):
    """
    Create a production-ready model using cold start evaluation setup.
    
    This function creates the final model that will be used for generating
    recommendations in the interactive demo.
    
    Args:
        ratings_df: User-restaurant ratings data
        user_profiles_df: User profile data
        restaurant_profiles_df: Restaurant profile data
    
    Returns:
        tuple: (model, user_id_map, restaurant_id_map, user_features, restaurant_features)
    """
    # Create cold start evaluation split (simulates production scenario)
    unique_users = ratings_df['userID'].unique()
    np.random.seed(42)
    cold_start_users = np.random.choice(unique_users, size=int(len(unique_users) * 0.2), replace=False)

    # Use 80% of users for training the production model
    train_ratings = ratings_df[~ratings_df['userID'].isin(cold_start_users)].copy()
    test_ratings = ratings_df[ratings_df['userID'].isin(cold_start_users)].copy()

    # Create comprehensive feature matrices
    user_features, user_id_map = create_user_features(user_profiles_df)
    restaurant_features, restaurant_id_map = create_restaurant_features(restaurant_profiles_df)

    # Create interaction matrices
    train_interactions = create_interaction_matrix(
        train_ratings, user_id_map, restaurant_id_map, use_weighted=True
    )
    test_interactions = create_interaction_matrix(
        test_ratings, user_id_map, restaurant_id_map, use_weighted=True
    )
    
    # Train production model with optimized hyperparameters
    print("Training production model...")
    model = LightFM(
        no_components=350,  # Latent factors for user/item representations
        loss='warp',        # WARP loss optimizes ranking metrics
        learning_schedule='adagrad',  # Adaptive learning rate
        learning_rate=0.05  # Initial learning rate
    )

    # Fit model with both collaborative and content-based information
    model.fit(
        interactions=train_interactions,
        user_features=user_features,
        item_features=restaurant_features,
        epochs=100,  # Number of training iterations
        verbose=True
    )
    
    return model, user_id_map, restaurant_id_map, user_features, restaurant_features

# ===== 9. RECOMMENDATION EXPLANATION SYSTEM =====

def explain_restaurant_recommendations(user_id, model, user_id_map, restaurant_id_map,
                                     restaurant_profiles_df, user_profiles_df,
                                     user_features, restaurant_features, ratings_df,
                                     n_recommendations=5):
    """
    Generate personalized restaurant recommendations with explanations.
    
    This function provides the core recommendation functionality:
    1. Validates user existence in the system
    2. Generates prediction scores for all restaurants
    3. Selects top-N recommendations
    4. Provides explanations for why each restaurant was recommended
    
    Args:
        user_id: ID of the user to generate recommendations for
        model: Trained LightFM model
        user_id_map: Mapping from user IDs to matrix indices
        restaurant_id_map: Mapping from restaurant IDs to matrix indices
        restaurant_profiles_df: Restaurant profile data
        user_profiles_df: User profile data
        user_features: User feature matrix
        restaurant_features: Restaurant feature matrix
        ratings_df: User-restaurant ratings data (for context)
        n_recommendations: Number of recommendations to generate
    
    Returns:
        dict: Structured recommendation results with explanations
    """
    # Validate that user exists in the system
    if user_id not in user_id_map:
        return {"error": f"User ID {user_id} not found in the system."}

    # Get user profile information for explanation generation
    user_profile = user_profiles_df[user_profiles_df['userID'] == user_id]
    if user_profile.empty:
        return {"error": f"User profile for User ID {user_id} not found."}
    user_profile = user_profile.iloc[0]

    # Get internal user index for model prediction
    user_idx = user_id_map[user_id]

    # Get total number of restaurants in the system
    n_items = restaurant_features.shape[0]

    # Create input arrays for batch prediction
    # Predict scores for this user against all restaurants
    user_array = np.full(n_items, user_idx)  # Repeated user index
    item_array = np.arange(n_items)          # All restaurant indices

    # Generate prediction scores using hybrid model
    # This combines collaborative filtering with content-based features
    scores = model.predict(
        user_ids=user_array,
        item_ids=item_array,
        user_features=user_features,
        item_features=restaurant_features
    )

    # Get indices of top-N recommendations (highest scores)
    top_item_indices = np.argsort(-scores)[:n_recommendations]

    # Create reverse mapping from matrix indices to restaurant IDs
    idx_to_restaurant_id = {idx: restaurant_id for restaurant_id, idx in restaurant_id_map.items()}

    # Build structured recommendation results
    recommendations = []

    for rank, idx in enumerate(top_item_indices):
        # Get restaurant ID and profile information
        restaurant_id = idx_to_restaurant_id[idx]
        restaurant = restaurant_profiles_df[restaurant_profiles_df['placeID'] == restaurant_id]

        if restaurant.empty:
            continue

        restaurant = restaurant.iloc[0]

        # Generate human-readable explanation for this recommendation
        explanation = generate_recommendation_explanation(
            user_profile, restaurant, scores[idx]
        )

        # Structure recommendation data
        recommendations.append({
            "rank": rank + 1,
            "restaurant_id": restaurant_id,
            "name": restaurant.get('name', f"Restaurant {restaurant_id}"),
            "dress code": restaurant.get('dress_code', 'Unknown'),
            "smoking area": restaurant.get('smoking_area', 'Unknown'),
            "cuisine": restaurant.get('Rcuisine', 'Unknown'),
            "price": restaurant.get('price', 'Unknown'),
            "city": restaurant.get('city', 'Unknown'),
            "explanation": explanation
        })

    return {
        "user_id": user_id,
        "total_recommendations": len(recommendations),
        "recommendations": recommendations
    }

def generate_recommendation_explanation(user_profile, restaurant, score):
    """
    Generate human-readable explanations for why a restaurant was recommended.
    
    This function analyzes the match between user preferences and restaurant
    attributes to create understandable explanations.
    
    Args:
        user_profile: Series containing user preference data
        restaurant: Series containing restaurant attribute data
        score: Model prediction score for this user-restaurant pair
    
    Returns:
        str: Human-readable explanation for the recommendation
    """
    reasons = []

    # Check budget preference alignment
    if 'budget' in user_profile and 'price' in restaurant:
        # Map user budget preferences to restaurant price ranges
        budget_price_map = {
            'low': ['low'],
            'medium': ['medium-low', 'medium'],
            'high': ['medium-high', 'high']
        }

        # Check if restaurant price matches user budget
        if restaurant['price'] in budget_price_map.get(user_profile['budget'], []):
            reasons.append(f"it matches your {user_profile['budget']} budget preference")

    # Check cuisine preference alignment
    # Note: Fixed typo in original code ('Rcusine' -> 'Rcuisine')
    if 'Rcuisine' in restaurant and 'Rcuisine' in user_profile:
        if restaurant['Rcuisine'] in user_profile['Rcuisine']:
            reasons.append(f"it offers {restaurant['Rcuisine']} cuisine as in your cuisine preference")

    # Check smoking preference alignment
    if 'smoker' in user_profile and 'smoking_area' in restaurant:
        # User prefers smoking-friendly venues
        if user_profile['smoker'] == 'true' and restaurant['smoking_area'] not in ['none', 'not permitted']:
            reasons.append("it has a smoking area")
        # User prefers non-smoking venues
        elif user_profile['smoker'] == 'false' and restaurant['smoking_area'] in ['none', 'not permitted']:
            reasons.append("it's a non-smoking restaurant")

    # Check dress code preference alignment
    if 'dress_preference' in user_profile and 'dress_code' in restaurant:
        # Match formal dress preferences
        if user_profile['dress_preference'] == 'formal' and restaurant['dress_code'] == 'formal':
            reasons.append(f"it matches your {restaurant['dress_code']} dress preference")
        # Match casual/informal dress preferences
        elif user_profile['dress_preference'] == 'casual' and restaurant['dress_code'] == 'informal':
            reasons.append(f"it matches your {restaurant['dress_code']} dress preference")
        elif user_profile['dress_preference'] == 'informal' and restaurant['dress_code'] == 'informal':
            reasons.append(f"it matches your {restaurant['dress_code']} dress preference")

    # Determine confidence level based on prediction score
    if score > 0.8:
        confidence = "strongly"
    elif score > 0.5:
        confidence = "moderately"
    else:
        confidence = "somewhat"

    # Construct final explanation
    if reasons:
        explanation = f"This restaurant is {confidence} recommended because {', and '.join(reasons)}."
    else:
        explanation = f"This restaurant is {confidence} recommended based on your preferences."

    return explanation

def interactive_recommendation_demo(model, user_id_map, restaurant_id_map,
                                  restaurant_profiles_df, user_profiles_df, ratings_df,
                                  user_features, restaurant_features):
    """
    Interactive command-line demo for testing the recommendation system.
    
    This function provides a user-friendly interface to:
    1. Enter user IDs to get recommendations
    2. Specify number of recommendations desired
    3. View detailed explanations for each recommendation
    4. Display user profile information for context
    
    Args:
        model: Trained LightFM model
        user_id_map: User ID to index mapping
        restaurant_id_map: Restaurant ID to index mapping
        restaurant_profiles_df: Restaurant profile data
        user_profiles_df: User profile data
        ratings_df: User-restaurant ratings data
        user_features: User feature matrix
        restaurant_features: Restaurant feature matrix
    """
    print("=== Restaurant Recommendation Demo ===")
    print(f"Available users: {list(user_id_map.keys())[:10]} (showing first 10)")

    # Main interaction loop
    while True:
        # Get user input
        user_input = input("\nEnter a user ID to get recommendations (or 'exit' to quit): ")

        # Check for exit condition
        if user_input.lower() == 'exit':
            break

        try:
            # Parse user ID (handle both string and integer IDs)
            user_id = int(user_input) if user_input.isdigit() else user_input
            
            # Get number of recommendations desired
            n_recs = int(input("How many recommendations would you like? (default 5): ") or "5")

            # Generate recommendations with explanations
            result = explain_restaurant_recommendations(
                user_id, model, user_id_map, restaurant_id_map,
                restaurant_profiles_df, user_profiles_df,
                user_features, restaurant_features, ratings_df,
                n_recommendations=n_recs
            )

            # Handle errors
            if "error" in result:
                print(f"Error: {result['error']}")
                continue

            # Display user profile information for context
            print(f"\n=== USER PROFILE FOR USER {user_id} ===")
            user_data = user_profiles_df[user_profiles_df['userID'] == user_id]
            if not user_data.empty:
                user_info = user_data.iloc[0]
                print(f"Cuisine Preference: {user_info.get('Rcuisine', 'Unknown')}")
                print(f"Budget: {user_info.get('budget', 'Unknown')}")
                print(f"Smoker: {user_info.get('smoker', 'Unknown')}")
                print(f"Dress Preference: {user_info.get('dress_preference', 'Unknown')}")
                print(f"Activity: {user_info.get('activity', 'Unknown')}")

            # Display recommendations
            print(f"\n=== TOP {result['total_recommendations']} RECOMMENDATIONS FOR USER {user_id} ===")
            for rec in result['recommendations']:
                print(f"\n{rec['rank']}. {rec.get('name', 'Restaurant ' + str(rec['restaurant_id']))}")
                print(f"   Cuisine: {rec['cuisine']}")
                print(f"   Price: {rec['price']}")
                print(f"   Dress Code: {rec['dress code']}")
                print(f"   Smoking Area: {rec['smoking area']}")
                print(f"   City: {rec.get('city', 'Unknown')}")
                print(f"   Explanation: {rec['explanation']}")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please check your input and try again.")

    print("Thank you for using the Restaurant Recommendation Demo!")

# ===== 10. SYSTEM INITIALIZATION AND DEMO =====

# Create production model with all components
print("Initializing production recommendation system...")
model, user_id_map, restaurant_id_map, user_features, restaurant_features = define_model_parameters(
    ratings, user_profiles, restaurant_profiles
)

print("Production model created successfully!")
print(f"System contains {len(user_id_map)} users and {len(restaurant_id_map)} restaurants")

# Launch interactive demonstration
print("Launching interactive recommendation demo...")
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