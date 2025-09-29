import logging
import os
import pickle
import pandas as pd
import azure.functions as func
from azure.storage.blob import BlobClient, __version__
import utils
import json

try:
    logging.info("Azure Blob Storage v" + __version__)

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
    logging.info('Azure Function appelée')

    try:
        req_body = req.get_json()
        user_id = req_body.get('id')
        rec_type = req_body.get('type')
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": f"Invalid JSON body: {e}"}),
            status_code=400,
            mimetype="application/json"
        )

    if not isinstance(user_id, int) or rec_type not in ["cb", "cf"]:
        return func.HttpResponse(
            json.dumps({
                "error": "Requête invalide. Format attendu: {id: int, type: 'cb' ou 'cf'}"
            }),
            status_code=400,
            mimetype="application/json"
        )

    try:
        if rec_type == "cb":
            recommended = utils.contentBasedRecommendArticle(
                articles_df, users_df, user_id)
        else:
            recommended = utils.collaborativeFilteringRecommendArticle(
                model, articles_df, users_df, user_id)

        return func.HttpResponse(
            json.dumps({"recommendations": recommended}, ensure_ascii=False),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Erreur pendant la recommandation: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
