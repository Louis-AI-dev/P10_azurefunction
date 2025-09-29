import logging
import os
import pickle
import pandas as pd
import azure.functions as func
from azure.storage.blob import BlobClient, __version__
import utils
import json

try:
    logging.info("CACAzure Blob Storage v" + __version__)

    connect_str = os.getenv('AzureWebJobsStorage')

    # Download blobs
    blob_articles = BlobClient.from_connection_string(
        conn_str=connect_str, container_name="filesforazurefunction", blob_name='articles_embeddings.pickle')
    blob_users = BlobClient.from_connection_string(
        conn_str=connect_str, container_name="filesforazurefunction", blob_name='users.pickle')
    blob_model = BlobClient.from_connection_string(
        conn_str=connect_str, container_name="filesforazurefunction", blob_name='model_svd.pickle')

    # Load pickles
    articles_df = pickle.loads(blob_articles.download_blob().readall())
    articles_df = pd.DataFrame(
        articles_df, columns=["embedding_" + str(i) for i in range(articles_df.shape[1])])
    users_df = pickle.loads(blob_users.download_blob().readall())
    model = pickle.loads(blob_model.download_blob().readall())

except Exception as ex:
    logging.error(f'Erreur de chargement des blobs: {ex}')
    articles_df, users_df, model = None, None, None


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Function recommend-article triggered.')

    try:
        req_body = req.get_json()
        logging.info(f"Request body: {req_body}")
    except ValueError as e:
        logging.error(f"Failed to parse JSON: {e}")
        return func.HttpResponse("Invalid JSON", status_code=400)

    id = req_body.get('id')
    type = req_body.get('type')

    logging.info(f"id={id}, type={type}")

    if isinstance(id, int) and isinstance(type, str):
        try:
            if type == "cb":
                recommended = utils.contentBasedRecommendArticle(articles_df, users_df, id)
            else:
                recommended = utils.collaborativeFilteringRecommendArticle(model, articles_df, users_df, id)

            logging.info(f"Recommendation result: {recommended}")
            return func.HttpResponse(str(recommended), status_code=200)

        except Exception as e:
            logging.error(f"Error while computing recommendation: {e}")
            return func.HttpResponse("Internal server error", status_code=500)

    else:
        logging.warning("Invalid request parameters.")
        return func.HttpResponse(
             "Requête invalide.\nDans le body doit figurer sous format json :\n- id (int)\n- type (cb ou cf)",
             status_code=400
        )
