import numpy as np
import pandas as pd

from scipy.sparse import csc_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors

import pickle
from scipy.sparse import save_npz, load_npz
from joblib import dump, load


class CoRecommender:
    """
    A class used to represent a Flexible Collaborative Filtering Recommendation System.

    This system can operate in either Item-Based or User-Based mode, allowing users to choose
    the recommendation approach that works best for their use case. Both approaches use a 
    Nearest Neighbors model with cosine similarity.

    Attributes
    ----------
    indic : str
        The name of the column in the input data that indicates the rating or interaction value.
    mode : str
        The recommendation mode, either 'item' for Item-Based or 'user' for User-Based.
    csr_mat_X : scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
        The sparse matrix representing the user-item interactions. Format depends on mode.
    user_mapper : dict
        A dictionary mapping user IDs to matrix indices.
    item_mapper : dict
        A dictionary mapping item IDs to matrix indices.
    user_inv_mapper : dict
        A dictionary mapping matrix indices back to user IDs.
    item_inv_mapper : dict
        A dictionary mapping matrix indices back to item IDs.
    model : sklearn.neighbors.NearestNeighbors
        The nearest neighbors model used to find similar items or users.
    is_model_data_loaded : bool
        A flag indicating whether the model data has been loaded from disk.
    all_rec_items_sorted_ids_dict : dict[int, list[str]]
        A dictionary storing sorted recommended item IDs.
    user_prev_data : set[str]
        A set containing the IDs of items the user has previously interacted with.

    Methods
    -------
    set_mode(mode) -> None
        Sets the recommendation mode (item-based or user-based).

    train_model(...) -> None
        Trains the recommendation system using the provided user-item interaction data.

    recommend_items(...) -> ...
        Recommends items to a user based on their previous interactions.

    print_pretty_recommendation(...) -> None
        Prints the recommended items in a user-friendly format.
    """

    def __init__(self, indic: str = 'rate', mode: str = 'item') -> None:
        """
        Initializes the CoRecommender with a specified indicator column name and mode.

        Parameters
        ----------
        indic : str, optional
            The name of the column in the input data that indicates the rating or interaction value (default is 'rate')
        mode : str, optional
            The recommendation mode, either 'item' for Item-Based or 'user' for User-Based (default is 'item')
        """
        self.__indic: str = indic
        self.__mode: str = mode.lower()
        if self.__mode not in ['item', 'user']:
            raise ValueError("Mode must be either 'item' or 'user'")

        # Matrix will be either CSC (for item-based) or CSR (for user-based)
        self.__matrix_X = None

        self.__user_mapper = dict()
        self.__item_mapper = dict()
        self.__user_inv_mapper = dict()
        self.__item_inv_mapper = dict()

        self.__model = None
        self.__is_model_data_loaded = False

        self.__all_rec_items_sorted_ids_dict: dict[int, list[str]] = dict()
        self.__user_prev_data: set[str] = set()

    def set_mode(self, mode: str) -> None:
        """
        Sets the recommendation mode.

        Parameters
        ----------
        mode : str
            The recommendation mode, either 'item' for Item-Based or 'user' for User-Based
        """
        mode = mode.lower()
        if mode not in ['item', 'user']:
            raise ValueError("Mode must be either 'item' or 'user'")

        self.__mode = mode
        # Reset model data when changing modes
        self.__is_model_data_loaded = False

    def get_mode(self) -> str:
        """
        Gets the current recommendation mode.

        Returns
        -------
        str
            The current recommendation mode ('item' or 'user')
        """
        return self.__mode

    def train_model(self, df: pd.DataFrame | dict) -> None:
        """
        Trains the recommendation system using the provided user-item interaction data.

        This method creates a sparse matrix from the input DataFrame and trains a
        Nearest Neighbors model using cosine similarity. The matrix orientation
        depends on the selected mode (item-based or user-based).

        Parameters
        ----------
        df : pd.DataFrame | dict
            The input DataFrame or Dict | JSON containing user-item interaction data. It must contain 'uid' for user IDs,
            'iid' for item IDs, and the interaction values in the column specified by `indic`.
        """

        if isinstance(df, dict):
            df = pd.DataFrame(df)

        self.__create_matrix(df)

        self.__model = NearestNeighbors(
            algorithm='brute',
            metric='cosine'
        )

        if self.__matrix_X is not None:
            self.__model.fit(self.__matrix_X)

        # Save model with mode-specific filename
        dump(self.__model,
             f'model_data/{self.__mode}_recommendations_model.joblib')

    def recommend_items(self,
                        user_id: str | None = None,
                        user_prev_data: pd.DataFrame | None = None,
                        n_recommendations: int = 10,
                        n_similar_entities: int = 5,
                        print_results: bool = True) -> dict:
        """
        Recommends items to a user based on the current mode.

        For item-based mode, this method identifies the most important items the user has interacted with 
        and finds similar items. For user-based mode, it finds similar users and recommends items they liked.

        Parameters
        ----------
        user_id : str, optional
            The user ID for user-based recommendations (required for user-based mode)
        user_prev_data : pd.DataFrame, optional
            A DataFrame containing the user's previous interactions with items (required for item-based mode)
        n_recommendations : int, optional
            The number of recommendations to return (default is 10)
        n_similar_entities : int, optional
            The number of similar items/users to consider (default is 5)
        print_results : bool, optional
            Whether to print the recommended items (default is True)

        Returns
        -------
        tuple
            Depending on the mode:
            - Item-based: a tuple containing a list of recommended item IDs and a dictionary mapping 
            each important item ID to a list of similar recommended item IDs
            - User-based: a list of recommended item IDs
        """
        if not self.__is_model_data_loaded:
            self.__load_model_data()

        if self.__mode == 'item':
            return self.__recommend_items_item_based(
                user_prev_data,  # type: ignore
                n_most_important_items=n_similar_entities,
                n_for_each_item=n_recommendations // n_similar_entities if n_similar_entities > 0 else 1,
                print_results=print_results
            )
        else:  # user-based
            return self.__recommend_items_user_based(
                user_id,  # type: ignore
                n_similar_users=n_similar_entities,
                n_recommendations=n_recommendations,
                print_results=print_results
            )

    def __save_model_data(self) -> None:
        """
        Saves the trained model and associated data to disk.
        """
        try:
            prefix = f"model_data/{self.__mode}"

            with open(f"{prefix}_user_mapper.pkl", 'wb') as file:
                pickle.dump(self.__user_mapper, file)

            with open(f"{prefix}_item_mapper.pkl", 'wb') as file:
                pickle.dump(self.__item_mapper, file)

            with open(f"{prefix}_user_inv_mapper.pkl", 'wb') as file:
                pickle.dump(self.__user_inv_mapper, file)

            with open(f"{prefix}_item_inv_mapper.pkl", 'wb') as file:
                pickle.dump(self.__item_inv_mapper, file)

            save_npz(f'{prefix}_matrix_X.npz', self.__matrix_X)

        except Exception as e:
            print(f"Error saving model data: {e}")

    def __load_model_data(self) -> None:
        """
        Loads the model and associated data from disk.
        """
        try:
            prefix = f"model_data/{self.__mode}"

            with open(f"{prefix}_user_mapper.pkl", 'rb') as file:
                self.__user_mapper = pickle.load(file)

            with open(f"{prefix}_item_mapper.pkl", 'rb') as file:
                self.__item_mapper = pickle.load(file)

            with open(f"{prefix}_user_inv_mapper.pkl", 'rb') as file:
                self.__user_inv_mapper = pickle.load(file)

            with open(f"{prefix}_item_inv_mapper.pkl", 'rb') as file:
                self.__item_inv_mapper = pickle.load(file)

            self.__matrix_X = load_npz(f'{prefix}_matrix_X.npz')
            self.__model = load(f'{prefix}_recommendations_model.joblib')
            self.__is_model_data_loaded = True

        except Exception as e:
            print(f"Error loading model data: {e}")
            self.__is_model_data_loaded = False

    def __create_matrix(self, df: pd.DataFrame) -> None:
        """
        Creates a sparse matrix from the user-item interaction data.

        The orientation of the matrix depends on the mode:
        - Item-based: Items as rows, users as columns (CSC matrix)
        - User-based: Users as rows, items as columns (CSR matrix)

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing user-item interaction data.
        """
        unique_users = np.unique(df['uid'])
        unique_items = np.unique(df['iid'])

        N = len(unique_users)  # number of unique users
        M = len(unique_items)  # number of unique items

        # Making list of range to the number of the users and items
        list_N = np.arange(N)
        list_M = np.arange(M)

        # Making dict hold a key of the user or item id and its "Index"
        self.__user_mapper = dict(zip(unique_users, list_N))
        self.__item_mapper = dict(zip(unique_items, list_M))

        # Making dict to map from index back to id
        self.__user_inv_mapper = dict(zip(list_N, unique_users))
        self.__item_inv_mapper = dict(zip(list_M, unique_items))

        # Get all the indexes corresponding to user or item ids
        user_index = [self.__user_mapper[i] for i in df['uid']]
        item_index = [self.__item_mapper[i] for i in df['iid']]

        if self.__mode == 'item':
            # Item-based: items as rows, users as columns
            self.__matrix_X = csc_matrix(
                (df[self.__indic], (item_index, user_index)),
                shape=(M, N)
            )
        else:  # user-based
            # User-based: users as rows, items as columns
            self.__matrix_X = csr_matrix(
                (df[self.__indic], (user_index, item_index)),
                shape=(N, M)
            )

        self.__save_model_data()

    def __find_similar_items(self, iid: str, n_neighbors: int, show_distance: bool = False):
        """
        Finds similar items to a given item using the trained model.

        Parameters
        ----------
        iid : str
            The item ID for which similar items are to be found.
        n_neighbors : int
            The number of similar items to return.
        show_distance : bool, optional
            Whether to return the distances along with the similar item IDs (default is False).

        Returns
        -------
        list[str]
            A list of similar item IDs.
        """
        if not self.__is_model_data_loaded:
            self.__load_model_data()

        # Get the item's vector (all users' interactions with this item)
        item_index = self.__item_mapper[iid]
        item_vector = self.__matrix_X[item_index]  # type: ignore
        item_vector = item_vector.reshape(1, -1)

        n_neighbors = min(n_neighbors, len(self.__item_mapper)) + 1

        neighbors = self.__model.kneighbors(  # type: ignore
            item_vector,
            n_neighbors=n_neighbors,
            return_distance=show_distance
        )

        neighbors_ids = list()
        for i in range(1, n_neighbors):  # Skip first (itself)
            n = neighbors.item(i)  # type: ignore
            neighbors_ids.append(self.__item_inv_mapper[n])

        return neighbors_ids

    def __find_similar_users(self, uid: str, n_neighbors: int, show_distance: bool = False):
        """
        Finds similar users to a given user using the trained model.

        Parameters
        ----------
        uid : str
            The user ID for which similar users are to be found.
        n_neighbors : int
            The number of similar users to return.
        show_distance : bool, optional
            Whether to return the distances along with the similar user IDs (default is False).

        Returns
        -------
        list[str]
            A list of similar user IDs.
        """
        if not self.__is_model_data_loaded:
            self.__load_model_data()

        # Get the user's vector (all items' interactions with this user)
        user_index = self.__user_mapper[uid]
        user_vector = self.__matrix_X[user_index]  # type: ignore
        user_vector = user_vector.reshape(1, -1)

        n_neighbors = min(n_neighbors, len(self.__user_mapper)) + 1

        neighbors = self.__model.kneighbors(  # type: ignore
            user_vector,
            n_neighbors=n_neighbors,
            return_distance=show_distance
        )

        neighbors_ids = list()
        for i in range(1, n_neighbors):  # Skip first (itself)
            n = neighbors.item(i)  # type: ignore
            neighbors_ids.append(self.__user_inv_mapper[n])

        return neighbors_ids

    def __recommend_items_item_based(
            self,
            user_prev_data: pd.DataFrame | dict,
            n_most_important_items: int = 10,
            n_for_each_item: int = 5,
            print_results: bool = True):
        """
        Recommends items to a user based on their previous interactions using item-based approach.

        Parameters
        ----------
        user_prev_data : pd.DataFrame
            A DataFrame containing the user's previous interactions with items.
        n_most_important_items : int, optional
            The number of most important items to consider (default is 10).
        n_for_each_item : int, optional
            The number of similar items to find for each important item (default is 5).
        print_results : bool, optional
            Whether to print the recommended items (default is True).

        Returns
        -------
        tuple
            A tuple containing a list of recommended item IDs and a dictionary mapping
            each important item ID to a list of similar recommended item IDs.
        """

        if isinstance(user_prev_data, dict):
            user_prev_data = pd.DataFrame(user_prev_data)

        if user_prev_data.empty:
            print('User has no previous preferences.')
            return dict(recommended_items_ids=[], relative_recommendations={})

        most_important_items = user_prev_data.nlargest(
            n_most_important_items, self.__indic)['iid']

        self.__user_prev_data = set(user_prev_data['iid'].values)
        self.__all_rec_items_sorted_ids_dict = dict()
        item_rec_dict = dict()

        if print_results:
            print(
                f'This user is interested in {len(most_important_items)} products.\n')

        for item_idx, item_id in enumerate(most_important_items):
            curr_similar_items_ids = self.__find_similar_items(
                item_id,
                n_neighbors=n_for_each_item
            )
            item_rec_dict[item_id] = curr_similar_items_ids

            if print_results:
                print((
                    f"{item_idx + 1}. Since you prefer product "
                    f"\"{item_id}\", you might also like these products:"
                ))

                for rec_item_idx, recommended_item_id in enumerate(curr_similar_items_ids):
                    found_before = self.__iid_is_found_in_rec(
                        recommended_item_id)
                    print(
                        f'{item_idx + 1}.{(rec_item_idx + 1):>2}. {recommended_item_id}\t{found_before}')

                print()

            self.__add_new_idd_to_rec(curr_similar_items_ids)

        recommended_items_ids = self.__combine_and_filter_rec_ids(
            self.__all_rec_items_sorted_ids_dict)

        relative_recommendations = item_rec_dict

        return dict(
            recommended_items_ids=recommended_items_ids,
            relative_recommendations=relative_recommendations
        )

    def __recommend_items_user_based(
            self,
            user_id: str,
            n_similar_users: int = 5,
            n_recommendations: int = 10,
            print_results: bool = True):
        """
        Recommends items to a user based on similar users' preferences.

        Parameters
        ----------
        user_id : str
            The user ID for which to make recommendations.
        n_similar_users : int, optional
            The number of similar users to consider (default is 5).
        n_recommendations : int, optional
            The number of recommendations to return (default is 10).
        print_results : bool, optional
            Whether to print the recommended items (default is True).

        Returns
        -------
        list[str]
            A list of recommended item IDs.
        """

        if user_id not in self.__user_mapper:
            print(f'User {user_id} not found in the dataset.')
            return []

        # Find similar users
        similar_users = self.__find_similar_users(
            user_id,
            n_neighbors=n_similar_users
        )

        if print_results:
            print(
                f'Found {len(similar_users)} similar users to userid "{user_id}".\n')

        # Get user's previous interactions
        user_idx = self.__user_mapper[user_id]
        user_vector = self.__matrix_X[user_idx].toarray(  # type: ignore
        ).flatten()  # type: ignore

        user_items = {
            self.__item_inv_mapper[i]: user_vector[i]
            for i in range(len(user_vector)) if user_vector[i] > 0
        }
        self.__user_prev_data = set(user_items.keys())

        # Get items liked by similar users
        recommendations = {}
        for sim_user_id in similar_users:
            sim_user_idx = self.__user_mapper[sim_user_id]
            sim_user_vector = self.__matrix_X[sim_user_idx].toarray(  # type: ignore
            ).flatten()  # type: ignore

            for item_idx in range(len(sim_user_vector)):
                # User has interacted with this item
                if sim_user_vector[item_idx] > 0:
                    item_id = self.__item_inv_mapper[item_idx]
                    if item_id not in self.__user_prev_data:  # Don't recommend items user already knows
                        if item_id in recommendations:
                            recommendations[item_id] += sim_user_vector[item_idx]
                        else:
                            recommendations[item_id] = sim_user_vector[item_idx]

        # Sort recommendations by score
        sorted_recs = sorted(
            recommendations.items(),
            key=lambda x: x[1], reverse=True)

        top_recs = [item_id for item_id,
                    score in sorted_recs[:n_recommendations]]

        if print_results:
            print(f"Top {len(top_recs)} recommendations for user {user_id}:")
            for i, item_id in enumerate(top_recs):
                print(f"{i+1}. {item_id}")
            print()

        return dict(
            recommended_items_ids=top_recs,
            similar_users_id=similar_users
        )

    @staticmethod
    def __combine_and_filter_rec_ids(all_rec_items_sorted_ids_dict: dict[int, list[str]]):
        """
        Combines and filters recommended item IDs, removing duplicates.

        Parameters
        ----------
        all_rec_items_sorted_ids_dict : dict[int, list[str]]
            A dictionary storing sorted recommended item IDs.

        Returns
        -------
        list[str]
            A list of unique recommended item IDs.
        """
        res_list = []
        sorted_idx_rate = list(all_rec_items_sorted_ids_dict.keys())
        for idx in sorted_idx_rate:
            rec_items = all_rec_items_sorted_ids_dict[idx]
            for rec_item in rec_items:
                if rec_item not in res_list:
                    res_list.append(rec_item)
        return res_list

    def __iid_is_found_in_rec(self, iid: str):
        """
        Checks if a given item ID is found in the user's previous data or in the current recommended items.

        Parameters
        ----------
        iid : str
            The item ID to check.

        Returns
        -------
        str
            A string indicating where the item ID was found, if at all.
        """
        if iid in self.__user_prev_data:
            return 'Found: User Prev Data'

        for iid_s in self.__all_rec_items_sorted_ids_dict.values():
            if iid in iid_s:
                return 'Found: RecSys Res'

        return ""

    def __add_new_idd_to_rec(self, curr_similar_items_ids: list[str]):
        """
        Adds new recommended item IDs to the recommendation list.

        Parameters
        ----------
        curr_similar_items_ids : list[str]
            A list of newly recommended item IDs.
        """
        for rec_item_idx, recommended_item_id in enumerate(curr_similar_items_ids):
            if recommended_item_id not in self.__user_prev_data:
                if self.__all_rec_items_sorted_ids_dict.get(rec_item_idx, None) is None:
                    self.__all_rec_items_sorted_ids_dict[rec_item_idx] = [
                        recommended_item_id]
                else:
                    if recommended_item_id not in self.__all_rec_items_sorted_ids_dict[rec_item_idx]:
                        self.__all_rec_items_sorted_ids_dict[rec_item_idx] \
                            .append(recommended_item_id)
