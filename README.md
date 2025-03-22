# CoRecommender

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/ibrahim-awny/)
[![Gmail](https://img.shields.io/badge/Gmail-Email-red?logo=gmail)](mailto:hima12awny@gmail.com)


A flexible collaborative filtering recommendation system that supports both item-based and user-based approaches using nearest neighbors with cosine similarity. Built with Python, NumPy, Pandas, and scikit-learn.

## Overview

CoRecommender is a powerful recommendation engine that combines both collaborative filtering approaches:

- **Item-Based Collaborative Filtering**: Recommends items similar to those a user has previously liked
- **User-Based Collaborative Filtering**: Recommends items that similar users have liked

The system is designed to be flexible, allowing you to switch between modes based on your specific use case and dataset characteristics.

## Features

- Dual-mode recommendation engine (item-based and user-based)
- Nearest neighbor model with cosine similarity for finding similar items/users
- Efficient sparse matrix implementation for handling large datasets
- Simple API for training and generating recommendations
- Customizable recommendation parameters
- Persistence of trained models and mapping dictionaries

## Installation

```bash
# Clone the repository
git clone https://github.com/hima12-awny/CoRecommender.git
cd CoRecommender

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- NumPy
- Pandas
- SciPy
- scikit-learn
- joblib

## Usage

### Basic Usage

```python
import pandas as pd
from co_recommender import CoRecommender

# Load your data (must contain 'uid', 'iid', and a rating/interaction column)
data = pd.read_csv('your_data.csv')
data = data.rename(columns={
    'user_id': 'uid',
    'item_id': 'iid',
    'rating': 'rate'
})

# Create a recommendation system instance
# 'mode' can be 'item' or 'user'
# 'indic' is the name of your rating/interaction column
rec_sys = CoRecommender(mode='item', indic='rate')

# Train the model
rec_sys.train_model(data)

# Get recommendations for a user
user_id = '12345'
user_history = data[data['uid'] == user_id][['iid', 'rate']]

recommendations = rec_sys.recommend_items(
    user_id=user_id,                   # Required for user-based mode
    user_prev_data=user_history,       # Required for item-based mode
    n_recommendations=10,              # Number of items to recommend
    n_similar_entities=5,              # Number of similar items/users to consider
    print_results=True                 # Whether to print verbose output
)

# Access recommended items
recommended_items = recommendations['recommended_items_ids']
```

### Input Data Format

Your input data must contain these columns:
1. `uid`: user ID who made the interaction
2. `iid`: item ID that the interaction was made on
3. A rating column (can be named anything, specified by the `indic` parameter)
   - This could represent ratings, clicks, views, etc.
   - Must be numeric values

Example:

| uid | iid | rate |
|-----|-----|------|
| 1   | 101 | 5.0  |
| 1   | 102 | 3.5  |
| 2   | 101 | 4.0  |
| 2   | 103 | 4.5  |

### Switching Modes

You can switch between item-based and user-based modes:

```python
# Switch to user-based collaborative filtering
rec_sys.set_mode('user')

# Switch to item-based collaborative filtering
rec_sys.set_mode('item')
```

### Recommendation Results

For item-based recommendations, the result contains:
- `recommended_items_ids`: List of all recommended item IDs
- `relative_recommendations`: Dictionary mapping each user-preferred item ID to similar recommended item IDs

For user-based recommendations, the result contains:
- `recommended_items_ids`: List of all recommended item IDs
- `similar_users_id`: List of all similar user IDs



## How It Works

CoRecommender uses a collaborative filtering approach based on nearest neighbors and cosine similarity:

1. **In item-based mode:**
   - The system creates a sparse matrix where items are rows and users are columns
   - For each item the user has interacted with, it finds similar items
   - It combines and ranks these similar items to create the final recommendations

2. **In user-based mode:**
   - The system creates a sparse matrix where users are rows and items are columns
   - It finds users similar to the target user
   - It identifies items those similar users liked but the target user hasn't interacted with yet

## Example

See the [usage example notebook](./usage_example.ipynb) for a complete demonstration with the MovieLens dataset.

## 📧 Contact

For questions or feedback, please open an issue on this repository.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/ibrahim-awny/)
[![Gmail](https://img.shields.io/badge/Gmail-Email-red?logo=gmail)](mailto:hima12awny@gmail.com)
