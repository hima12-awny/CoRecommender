#!/usr/bin/env python
# coding: utf-8

# # CoRecommender - Usage Example 
# A flexible collaborative filtering recommendation system that supports both item-based and user-based approaches using nearest neighbors with cosine similarity. Built with Python, NumPy, Pandas, and scikit-learn.

# In[ ]:


import numpy as np
import pandas as pd
from co_recommender import CoRecommender


# ## Read Data Sources

# the data must contains:
# 1. `uid`: user id, how made the interaction
# 2. `iid`: item id, that the interaction made on it.
# 3. `dynamic`: this user free column name, act as indicator of the interaction and it's scale means how mush user interacts with this item.
# 
# `dynamic` means, you can name it whatever you want, it could be anything, but it should be Numeric Value.
# it could be for example rate, num_of_visits, likes, num_of_clicks...

# #### Read it from CSV

# In[3]:


users_interaction_data = pd.read_csv(
    'database/ratings.csv',
    usecols=['userId', 'movieId', 'rating'],
).rename(
    columns={
        'userId': 'uid',
        'movieId': 'iid',
        'rating': 'rate'
    })

movies = pd.read_csv(
    'database/movies.csv',
    index_col='movieId',
    converters={"genres": lambda x: ', '.join(sorted(x.split('|')))})


# #### Read it from JSON

# In[29]:


import json
with open('database/users_interaction_data_sample.json', 'r') as f:
    users_interaction_data_json = f.read()
    users_interaction_data = json.loads(users_interaction_data_json)
    users_interaction_data = pd.DataFrame(users_interaction_data)


# In[30]:


print(users_interaction_data_json)


# #### Preview

# In[23]:


users_interaction_data


# In[5]:


movies.head()


# ## Create CoRecommender
# you need to specify
# 1. `mode`: the tool handle two recommendations types, item-based and user-based, you should choose one type at once, <br> `user` -> user-based <br> `item` -> item-based.
# 2. `indic`: indication column name, `dynamic` colname like above, ex. rate. 

# In[6]:


recommendation_type = 'item'
interaction_indication_col = 'rate'

rec_sys = CoRecommender(
    mode=recommendation_type,
    indic=interaction_indication_col,
)


# ### Train The model
# Means Convert the user-item interaction data into suitable formate and save it based on recommendation Type

# In[7]:


rec_sys.train_model(users_interaction_data)


# ### Running Sample

# In[8]:


# get unique users and items
uids = users_interaction_data['uid'].unique()
rec_data = users_interaction_data


# In[9]:


# get random user id for testing the model.
user_id = np.random.choice(uids)

# get all data for a specific user only.
user_preferred_products = rec_data[
    rec_data['uid'] == user_id][['iid', 'rate']]

user_preferred_products = user_preferred_products.sort_values(
    by='rate', ascending=False).reset_index(drop=True)

print(f'User id: {user_id}, Preferred Items:')
user_preferred_products.head()


# ### Get Recommendations
# 
# #### Parameters
# 1. `user_id`: that you want to get recommendation for him, and based on his behavior. `required` for user-based.
# 2. `user_prev_data`: user previous interaction data, that based on it get recommendations. `required` for item-based
# 3. `n_recommendations`: max number of recommendations items you want to get `required`.
# 4. `n_similar_entities`: for each item or user how much you wan to get similar items. `required`
# 5. `print_results`: print logs, `optional` with defaults `True`
# 
# #### Returns
# in Case User-Based
# 1. `recommended_items_ids`: list of all recommended items ids.
# 2. `similar_users_id`:  list of all similar users ids.
# 
# in Case Item-Based
# 1. `recommended_items_ids`: list of all recommended items ids.
# 2. `relative_recommendations`: this dictionary that contains the user preferred item id as key, and similar items ids that recommended based on this key.

# In[ ]:


# NOTE: you can't call or use this method before train the mode at least one time.
recommendations = rec_sys.recommend_items(
    user_id=user_id,  # user id to recommend items for.
    user_prev_data=user_preferred_products,  # user previous interactions.
    n_recommendations=200,  # number of items to recommend.
    n_similar_entities=10,  # number of similar items or users to get.
    print_results=False  # set to True if you want to print the results.
)


# ### Show Relative Recommendations Results (Item-based)

# In[11]:


if recommendation_type == 'item':
    user_watched_movies = set(user_preferred_products['iid'].values)
    print("Relative Recommendations:")

    relative_recs = recommendations['relative_recommendations']

    printed_mids = set()

    for i, (r_mid, r_mids) in enumerate(relative_recs.items(), 1):

        print(f'{i}. because user loved ->', ' - '.join(
            movies.loc[r_mid][['title', 'genres']].values))

        e = 0
        for m_id in r_mids:
            if m_id in user_watched_movies or m_id in printed_mids:
                continue

            printed_mids.add(m_id)
            e = e+1
            print(f'   {e}.', ' - '.join(
                movies.loc[m_id][['title', 'genres']].values))

        print('-'*100)


# ### User Preferred Items 

# In[12]:


print("Top 10 movies rated by user: ")
top_10_movies = user_preferred_products.sort_values(
    'rate', ascending=False).head(10)

movies.merge(top_10_movies, left_index=True, right_on='iid')[
    ['title', 'genres', 'rate']].reset_index(drop=True)


# ### Recommended Items

# In[13]:


print("Recommended Movies:")

rec_movies = movies.loc[recommendations['recommended_items_ids']]
rec_movies.head(10)

