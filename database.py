import os
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "jevaraat")
PRODUCT_COLLECTIONS = ("goldbars", "necklaces", "ladiesrings", "rings")


def get_database() -> Any:
	"""Return the configured MongoDB database."""
	if not MONGO_URI:
		raise RuntimeError("MONGO_URI is missing from the environment")

	client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
	client.admin.command("ping")
	return client[MONGO_DB_NAME]


def fetch_product_documents() -> list[dict[str, Any]]:
	"""Fetch all products from the collections used by the Jevaraat app."""
	database = get_database()
	products: list[dict[str, Any]] = []

	for collection_name in PRODUCT_COLLECTIONS:
		for product in database[collection_name].find({}):
			products.append({"collection": collection_name, "product": product})

	return products