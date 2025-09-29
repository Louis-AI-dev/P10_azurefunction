from math import *
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from heapq import nlargest

# Recommandation Content-Based
def contentBasedRecommendArticle(articles, users, user_id, n=5):

    articles_read = users['click_article_id'].loc[user_id]

    if len(articles_read) == 0:
        return "L'utilisateur n'a lu aucun article"

    articles_read_embedding = articles.loc[articles_read]

    articles = articles.drop(articles_read)

    matrix = cosine_similarity(articles_read_embedding, articles)

    rec = []

    for i in range(n):
        coord_x = floor(np.argmax(matrix)/matrix.shape[1])
        coord_y = np.argmax(matrix)%matrix.shape[1]

        rec.append(int(coord_y))

        matrix[coord_x][coord_y] = 0

    return rec

# Recommandation Collaborative Filtering
def collaborativeFilteringRecommendArticle(model, articles, users, user_id, n=5):

    index = list(articles.index)

    articles_read = users['click_article_id'].loc[user_id]

    for ele in articles_read:
        if ele in index:
            index.remove(ele)

    results = dict()

    for i in index:
        pred = model['algo'].predict(user_id, i)
        results[pred.iid] = pred.est
    
    return nlargest(n, results, key = results.get)