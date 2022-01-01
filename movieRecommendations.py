# Run in an anaconda enviroment with python 3.9.7 and numpy, pandas and scipy installed.
# Change the FILE_LOC-variable to the folder where the movielens files are located

from collections import Counter
import numpy
import pandas as pd
from scipy import stats


GROUP_OF_USERS = [5, 254, 35]
NUMBER_OF_SEQUENCES = 5
MOVIES_TO_RECOMMEND = 20
FILE_LOC = "x"
WHY_NOT_ID = 3475
LOWERID = 940
HIGHERID = 333
GENRE_NAME = "Animation"
MOVIES_TO_RECOMEND = 20

# calculate the pearson correlation between one user and others, and return a list with them sorted in descending order according to correlation
def calc_user_similarities(ratings, user):
    correlations = []
    user_data = ratings[ratings["userId"] == user]

    for uid in pd.unique(ratings["userId"]):
        if uid == user:
            continue
        else:
            other_user_data = ratings.loc[ratings["userId"] == uid]

            # use only the movies both users have rated
            other_user_data = other_user_data.loc[other_user_data["movieId"].isin(user_data["movieId"])]
            user_common_movies = user_data.loc[user_data["movieId"].isin(other_user_data["movieId"])]

            # if the users have less than 5 movies in common, they are assumed to be dissimilar
            if len(user_common_movies.index) < 5:
                continue
            else:
                cor = stats.pearsonr(user_common_movies["rating"], other_user_data["rating"])[0]
                correlations.append([uid, cor])

    correlations.sort(reverse=True, key=lambda x: x[1])

    return correlations

# iterate through all the movies not seen by the user, calculate a score for them, and return list of tuples of movieids and their scores
def predict_from_users(ratings, correlations, user):
    movie_scores = []
    user_data = ratings.loc[ratings["userId"] == user]
    user_mean = user_data["rating"].mean()
    
    not_seen = ratings.loc[~ratings["movieId"].isin(user_data["movieId"])]

    for movie in pd.unique(not_seen["movieId"]):
        ratings_of_movie = not_seen.loc[not_seen["movieId"] == movie]
        # the similarity values of the users who have seen the movie being scored
        seen_similarities = correlations.loc[correlations["uid"].isin(ratings_of_movie["userId"])]
        # Choose the 20 most similar users as the neighbourhood
        most_similar = seen_similarities.sort_values("cor", ascending=False).head(20)
        sum_of_sims = most_similar["cor"].sum()
        if sum_of_sims == 0:
            continue
        else:
            differences_from_mean = numpy.zeros(20)
            for j, neighbour in enumerate(most_similar["uid"]):
                neighbour_ratings = ratings.loc[ratings["userId"] == neighbour]
                neighbour_mean = neighbour_ratings["rating"].mean()
                # Get the correlation between the users
                sim = most_similar.loc[most_similar["uid"] == neighbour].cor.item()
                # Get the rating the neighbour has given for the movie
                rating = neighbour_ratings.loc[neighbour_ratings["movieId"] == movie].rating.item()
                difference = sim * (rating - neighbour_mean)
                differences_from_mean[j] = difference
            score = user_mean + ((differences_from_mean.sum())/sum_of_sims)
            movie_scores.append([movie, score])

    movie_scores.sort(reverse=True, key=lambda x: x[1])

    return movie_scores


def index_recs(list_of_relevances):
    # Get all the movie id's that have prediction scores
    all_movies = list_of_relevances[0].index.union(list_of_relevances[1].index.union(list_of_relevances[2].index))

    # Create an array which contains the rank each movie has for each user
    movie_ranks = numpy.zeros((all_movies.shape[0], 3))
    for index, movie in enumerate(all_movies):
        for user in range(3):
            try:
                movie_ranks[index, user] = list_of_relevances[user].index.get_loc(movie)
            except KeyError:
                # if a user does not have a score for a movie
                movie_ranks[index, user] = numpy.nan

    # Create a dataframe out of the ranks, and drop those movies for which all users don't have a score
    df_of_ranks = pd.DataFrame(data=movie_ranks, index=all_movies)
    df_of_ranks.dropna(inplace=True)

    # Give each movie a score based on the median rank and max rank of it
    df_of_ranks["median"] = df_of_ranks.median(axis=1)
    df_of_ranks["max"] = df_of_ranks.max(axis=1)
    df_of_ranks["med+max"] = df_of_ranks["median"] + df_of_ranks["max"]

    return df_of_ranks.sort_values(by=["med+max"])

# Function that prints out information about why a movie was not recommended
def why_not_movie(movie_id, rel_scores, index_scores):
    if movie_id not in rel_scores.index:
        print("None of the users got a recommendation score for this movie, so it couldn't be recommended to the group")
    else:
        movie_scores = rel_scores.loc[movie_id]
        if movie_scores.isnull().values.any():
            print("The movie was not recommended to the group because at least one of the users didn't have a recommendation score for it")
        else:
            if index_scores.index.get_loc(movie_id) < MOVIES_TO_RECOMMEND:
                print("The movie was recommended!")
            else:
                movie_rank = index_scores.index.get_loc(movie_id)
                if movie_rank < MOVIES_TO_RECOMMEND + 30:
                    print(f"The movie was ranked number {movie_rank}, so it would have been recommended had the user asked for a slightly larger amount of recommendations")
                else:
                    check_locs(movie_id, index_scores)

# Helper function for why_not_movie that prints more precise info about the movies rank and the reasons for it
def check_locs(movie_id, index_scores):
    locs = index_scores.copy()
    movie_loc = locs.index.get_loc(movie_id)

    # How high a movie should be to be in the top 10% of the recommendations
    top_limit = len(index_scores.index) / 10

    if movie_loc < top_limit:
        print("The movie was in the top 10%% of recommendations by my own method")
    else:
        print("The movie wasn't in the top 10%% of movies by my own method")
    
    # Get the movies location for both the median user and the one that liked it least
    median_loc = locs.sort_values(by="median").index.get_loc(movie_id)
    lowest_loc = locs.sort_values(by="max").index.get_loc(movie_id)


    # Additional info about the movies location
    if median_loc < MOVIES_TO_RECOMMEND:
        print("The movie would have been recommended to the median user, but one user disliked it too much")
    elif lowest_loc < MOVIES_TO_RECOMMEND:
        print("The movie would have been recommended by least misery, but the median user didn't like it enough")
    elif median_loc < top_limit and lowest_loc < top_limit:
        print("Both the median user and the one that liked the movie the least liked it somewhat, but not enough for it to be recommended")
    elif median_loc < top_limit:
        print("The median user liked the movie quite a bit, but one user disliked it")
    elif lowest_loc < top_limit:
        print("Everybody liked the movie somewhat, but nobody really liked it")
    else:
        print("Nobody really liked the movie that much")
    
# A function that tells why one movie was higher than the other one
def why_not_higher(lower_id, higher_id, index_scores, rel_scores, lower_title, higher_title):
    scores = index_scores.copy()

    if lower_id not in scores.index and higher_id not in scores.index:
        print("Neither movie was recommended at all. Printing explanations:")
        why_not_movie(lower_id, rel_scores, scores)
        why_not_movie(higher_id, rel_scores, scores)
    elif lower_id not in scores.index:
        print(f"Movie {lower_title} was not recommended at all. Printing an explanation:")
        why_not_movie(lower_id, rel_scores, scores)
    elif higher_id not in scores.index:
        print(f"Movie {higher_title} was not recommended at all. Printing an explanation:")
        why_not_movie(higher_id, rel_scores, scores)
    else:
        if scores.index.get_loc(lower_id) < scores.index.get_loc(higher_id):
            print(f"Movie {lower_title} is higher than {higher_title}, check the parameters!")
        else:
            # Get the movies locations
            lower_median = scores.sort_values(by=["median"]).index.get_loc(lower_id)
            lower_lowest = scores.sort_values(by=["max"]).index.get_loc(lower_id)

            higher_median = scores.sort_values(by=["median"]).index.get_loc(higher_id)
            higher_lowest = scores.sort_values(by=["max"]).index.get_loc(higher_id)

            if higher_median < lower_median and higher_lowest < lower_lowest:
                print(f"Both the median user and the least happiest user were predicted to like the movie {higher_title} better")
            elif higher_median < lower_median:
                print(f"Least misery would have recommended {lower_title}, but the median user preferred {higher_title} too much")
            elif higher_lowest < lower_lowest:
                print(f"{lower_title} would have been recommended to the median user, but it was too disliked by one user")
            else:
                print(f"{lower_title} should have been higher, the system has a bug!")

# Explain why a genre was not recommended to the group
def why_not_genre(genre_name, index_scores, movie_info):
    scores = index_scores.copy()
    movie_info = movie_info.copy()

    group_recomendations = scores.head(MOVIES_TO_RECOMMEND)
    median_recomendations = scores.sort_values(by="median").head(MOVIES_TO_RECOMMEND)
    least_misery_recomendations = scores.sort_values(by="max").head(MOVIES_TO_RECOMMEND)

    # Get the genres of the movies recommended to the group, by median aggregation and by least misery aggregation

    group_genres = []
    for movie in group_recomendations.index:
        group_genres.extend(movie_info[movie_info["movieId"] == movie].genres.item().split("|"))

    median_genres = []
    for movie in median_recomendations.index:
        median_genres.extend(movie_info[movie_info["movieId"] == movie].genres.item().split("|"))

    least_misery_genres = []
    for movie in least_misery_recomendations.index:
        least_misery_genres.extend(movie_info[movie_info["movieId"] == movie].genres.item().split("|"))

    # Calculate how many times on average a genre that was recommended appeared, and how many times each genre appeared

    group_counts = Counter(group_genres)
    group_avg = numpy.mean(list(group_counts.values()))

    median_counts = Counter(median_genres)
    median_avg = numpy.mean(list(median_counts.values()))

    least_misery_counts = Counter(least_misery_genres)
    least_misery_avg = numpy.mean(list(least_misery_counts.values()))

    # How many times the genre of interest was recommended
    group_genre_recs = group_counts[genre_name]
    median_genre_recs = median_counts[genre_name]
    least_misery_genre_recs = least_misery_counts[genre_name]

    if group_genre_recs > group_avg:
        print("The genre was one of the genres recommended most!")
    elif median_genre_recs > median_avg and least_misery_genre_recs > least_misery_avg:
        print("The genre was recommended to both the median user and the most miserable user, but not to the group. Likely a bug")
    elif median_genre_recs > median_avg:
        print("The genre was recommended often to the median user, but not to the group because one user didn't like it")
    elif least_misery_genre_recs > least_misery_avg:
        print("The genre was recommended to one user, but the others weren't interested")
    else:
        print("None of the users were interested in the genre")

# Sort the dataframe given as a parameter in descending order according to the median value of each row
def median_aggregation(rel_scores):
    med_scores = rel_scores.copy()
    med_scores["med"] = med_scores.median(axis=1)
    med_scores.sort_values(by=["med"], inplace=True, ascending=False)
    return med_scores

# Sort the dataframe given as a parameter in descending order according to the minimum value of each row
def least_misery_aggregation(rel_scores):
    min_scores = rel_scores.copy()
    min_scores["min"] = min_scores.min(axis=1)
    min_scores.sort_values(by=["min"], inplace=True, ascending=False)
    return min_scores

# Get a dataframe containing the scores each user has gotten for each movie sorted by their score for the whole grouo
# and calculate the satisfaction difference between the most satisfied and the least satisfied user
def calc_sat_diff(df_of_scores):
    satisfactions = []
    scores = df_of_scores.copy()
    # The k-items recomended to the group
    group_list = scores.head(MOVIES_TO_RECOMEND)

    for user in range(len(GROUP_OF_USERS)):
        # calculate the score the user gets from the group list
        group_list_sat = group_list.iloc[:, user].sum()

        # Calculate the score the user gets from the top k-items for themselves
        scores.sort_values(by=scores.columns[user], inplace=True, ascending=False)
        user_list = scores.head(MOVIES_TO_RECOMEND)
        user_list_sat = user_list.iloc[:, user].sum()

        user_sat = group_list_sat/user_list_sat
        satisfactions.append(user_sat)

    satisfaction_diff = max(satisfactions) - min(satisfactions)
    return satisfaction_diff


# Recomend sequantial lists of movies to a group of users according to a hybrid aggregation model
# using both median and least misery aggregation
def sequence_recomender(rel_scores, movie_names):
    seq_scores = rel_scores.copy()
    # weight given to the least misery aggregation
    alpha = 0

    for seq in range(NUMBER_OF_SEQUENCES):
        if alpha == 0:
            # For the first round, only use median aggregation
            seq_scores.insert(len(seq_scores.columns), "hybrid_score", median_aggregation(seq_scores)["med"])
        else:
            # Do the median and least misery aggregations, weight the scores given by them with alpha and add them together
            median_scores = median_aggregation(seq_scores)["med"].multiply(1-alpha)
            min_scores = least_misery_aggregation(seq_scores)["min"].multiply(alpha)
            hybrid_scores = median_scores.add(min_scores)

            seq_scores.insert(len(seq_scores.columns), "hybrid_score", hybrid_scores)
        
        seq_scores.sort_values(by=["hybrid_score"], inplace=True, ascending=False)

        print(f"The most relevant movies for the group in sequence {seq + 1} are:")
        for i in range(MOVIES_TO_RECOMEND):
            print(movie_names[movie_names["movieId"] == seq_scores.index[i]].title.item())
        print()

        # Calculate the alpha for the next sequence
        alpha = calc_sat_diff(seq_scores)

        # delete the scores for this iteration so that they do not affect calculation in the next iteration
        seq_scores.drop(columns=["hybrid_score"], inplace=True)
        # drop the movies that the group was recomended
        seq_scores.drop(seq_scores.index[:MOVIES_TO_RECOMEND], inplace=True)
    

if __name__ == "__main__":
    df_of_ratings = pd.read_csv(FILE_LOC + "ratings.csv")
    movies = pd.read_csv(FILE_LOC + "movies.csv")
    
    all_relevances = []
    for u in GROUP_OF_USERS:
        sims = calc_user_similarities(df_of_ratings, u)
        sims_df = pd.DataFrame(sims, columns=["uid", "cor"])
        relevances = numpy.asanyarray(predict_from_users(df_of_ratings, sims_df, u))
        relevances = pd.Series(data=relevances[:, 1], index=relevances[:, 0])
        all_relevances.append(relevances)

    # Make a pandas dataframe where each row contains the relevance scores the users have gotten for a particular movie
    relevance_scores = pd.concat(all_relevances, axis=1)
    relevance_scores.columns = GROUP_OF_USERS
    

    own_scores = index_recs(all_relevances)
    print("The most relevant movies for the group according to my own method are:")
    for i in range(MOVIES_TO_RECOMMEND):
        recommended_movie = movies[movies["movieId"] == own_scores.index[i]].title.item()
        movie_genres = movies[movies["movieId"] == own_scores.index[i]].genres.item()
        print(f"{recommended_movie} genres: {movie_genres}")

    print()

    why_not_name = movies[movies["movieId"] == WHY_NOT_ID].title.item()
    print(f"Why wasn't movie {why_not_name} recommended?")
    why_not_movie(WHY_NOT_ID, relevance_scores, own_scores)

    print()

    higher_name = movies[movies["movieId"] == HIGHERID].title.item()
    lower_name = movies[movies["movieId"] == LOWERID].title.item()
    print(f"Why is movie {higher_name} higher than {lower_name}")
    why_not_higher(LOWERID, HIGHERID, own_scores, relevance_scores, lower_name, higher_name)

    print()

    print(f"Why isn't genre {GENRE_NAME} not recommended much?")
    why_not_genre(GENRE_NAME, own_scores, movies)
