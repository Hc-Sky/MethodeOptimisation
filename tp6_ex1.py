from pymongo import MongoClient

# Connexion
client = MongoClient('mongodb://localhost:27017/')
db = client.local
books = db.books

def execute_aggregation(title, pipeline):
    print(f"\n{title}")
    print("-" * len(title))
    results = books.aggregate(pipeline)
    for result in results:
        print(result)

# 1. Nombre total de livres
execute_aggregation(
    "1. Nombre total de livres dans la collection",
    [{"$group": {"_id": None, "totalBooks": {"$sum": 1}}}]
)

# 2. Nombre total de pages
execute_aggregation(
    "2. Nombre total de pages de tous les livres",
    [{"$group": {"_id": None, "totalPages": {"$sum": "$pageCount"}}}]
)

# 3. Nombre moyen de pages par catégorie
execute_aggregation(
    "3. Nombre moyen de pages par catégorie",
    [
        {"$unwind": "$categories"},
        {"$group": {"_id": "$categories", "avgPages": {"$avg": "$pageCount"}}},
        {"$sort": {"avgPages": -1}}
    ]
)

# 4. Les 3 meilleurs auteurs
execute_aggregation(
    "4. Les 3 meilleurs auteurs avec le plus de livres",
    [
        {"$unwind": "$authors"},
        {"$group": {"_id": "$authors", "bookCount": {"$sum": 1}}},
        {"$sort": {"bookCount": -1}},
        {"$limit": 3}
    ]
)

# 5. Livres publiés par année
execute_aggregation(
    "5. Nombre de livres publiés chaque année",
    [
        {"$project": {
            "year": {
                "$cond": [
                    {"$eq": [{"$type": "$publishedDate"}, "string"]},
                    {"$year": {"$dateFromString": {"dateString": "$publishedDate", "format": "%Y-%m-%dT%H:%M:%S.%LZ"}}},
                    null
                ]
            },
            "_id": 0
        }},
        {"$match": {"year": {"$ne": null}}},
        {"$group": {"_id": "$year", "bookCount": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
)

# 6. Statistiques sur les catégories
execute_aggregation(
    "6. Nombre moyen et maximum de catégories par livre",
    [
        {"$project": {
            "categoryCount": {"$size": {"$ifNull": ["$categories", []]}}
        }},
        {"$group": {
            "_id": None,
            "avgCategories": {"$avg": "$categoryCount"},
            "maxCategories": {"$max": "$categoryCount"}
        }}
    ]
)

# Fermeture de la connexion
client.close()