import logging
import os
import pickle
import pandas as pd
import azure.functions as func
from azure.storage.blob import BlobClient,  __version__
import utils

try:
    logging.info("Azure Blob Storage v" + __version__)

    connect_str = os.getenv('AzureWebJobsStorage')

    ##### Chargement des fichiers #####
    # Download blobs
    blob_articles = BlobClient.from_connection_string(conn_str=connect_str, container_name="filesforazurefunction", blob_name='articles_embeddings.pickle')
    blob_users = BlobClient.from_connection_string(conn_str=connect_str, container_name="filesforazurefunction", blob_name='users.pickle')
    blob_model = BlobClient.from_connection_string(conn_str=connect_str, container_name="filesforazurefunction", blob_name='model_svd.pickle')

    # Load to pickle
    articles_df = pickle.loads(blob_articles.download_blob().readall())
    articles_df = pd.DataFrame(articles_df, columns=["embedding_" + str(i) for i in range(articles_df.shape[1])])

    users_df = pickle.loads(blob_users.download_blob().readall())

    model = pickle.loads(blob_model.download_blob().readall())

except Exception as ex:
    print('Exception:')
    print(ex)


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req_body = req.get_json()
    except ValueError:
        pass
    else:
        id = req_body.get('id')
        type = req_body.get('type')

    if isinstance(id, int) and isinstance(type, str):

        recommended = utils.contentBasedRecommendArticle(articles_df, users_df, id) if type == "cb" else utils.collaborativeFilteringRecommendArticle(model, articles_df, users_df, id)

        return func.HttpResponse(str(recommended), status_code=200)

    else:
        return func.HttpResponse(
             "Requête invalide.\nDans le body doit figurer sous format json :\n- id (int)\n- type (cb ou cf)",
             status_code=400
        )