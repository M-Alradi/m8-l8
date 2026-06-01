"""Module 8 — Applied Lab: Vector Retrieval.

Implement BM25, dense, and hybrid retrievers against a Weaviate index of the
CQADupStack + Stack Exchange technical-Q&A corpus, then evaluate all three on
the bundled 60-pair labeled set.

Methodology (canonical — autograder enforces):
- recall@k: gold_doc_id in top_k_returned_ids; mean over all queries.
- MRR: 1-indexed position of gold_doc_id in returned list of length 10;
  1/rank if found, 0 if not; mean over all queries.
- Hybrid alpha: 0.5 for the base assignment.
- Top-k for retrieval calls during evaluation: k=10; recall@5 is the top-5
  slice of those 10. One retrieval call per query.
"""

import json
from typing import Callable
from collections import defaultdict

from sentence_transformers import SentenceTransformer
import weaviate

CLASS_NAME = "Post"


def create_schema(client: weaviate.Client) -> None:
    """Create the Post class in Weaviate.

    Properties:
      - doc_id (text, filterable): globally-unique "{subset}:{post_id}"
      - subset (text, filterable): one of "programmers" / "webmasters" / "android"
      - title (text, BM25-indexed)
      - question_text (text, BM25-indexed)
      - answer_text (text, BM25-indexed)
      - text (text, stored — NOT BM25-indexed; double-counts otherwise)

    Class-level config:
      - vectorizer: "none" (we supply vectors externally)
      - vectorIndexConfig: {"distance": "cosine"}

    If the class already exists, delete it first (so re-running create_schema
    on an existing index is idempotent).

    The BM25 retrieval surface is the three BM25-indexed properties; `text`
    exists as the unified dense-embedding source and a backward-compat
    "full doc" view but does not participate in BM25.
    """
    existing = client.schema.get()
    class_names = {c["class"] for c in existing.get("classes", [])}

    if CLASS_NAME in class_names:
        client.schema.delete_class(CLASS_NAME)

    class_def = {
        "class": CLASS_NAME,
        "vectorizer": "none",
        "vectorIndexConfig": {
            "distance": "cosine"
        },
        "properties": [
            {
                "name": "doc_id",
                "dataType": ["text"],
                "indexSearchable": False,
                "indexFilterable": True,
                "tokenization": "field",
            },
            {
                "name": "subset",
                "dataType": ["text"],
                "indexSearchable": False,
                "indexFilterable": True,
                "tokenization": "field",
            },
            {
                "name": "title",
                "dataType": ["text"],
                "indexSearchable": True,
                "indexFilterable": False,
                "tokenization": "word",
            },
            {
                "name": "question_text",
                "dataType": ["text"],
                "indexSearchable": True,
                "indexFilterable": False,
                "tokenization": "word",
            },
            {
                "name": "answer_text",
                "dataType": ["text"],
                "indexSearchable": True,
                "indexFilterable": False,
                "tokenization": "word",
            },
            {
                "name": "text",
                "dataType": ["text"],
                "indexSearchable": False,
                "indexFilterable": False,
                "tokenization": "word",
            },
        ],
    }

    client.schema.create_class(class_def)


def index_corpus(client: weaviate.Client, corpus_path: str, embedder) -> int:
    """Embed and ingest the corpus into the Post class.

    For each line in `corpus_path` (JSONL, one document per line):
      - Embed `row["text"]` with `embedder.encode(...)` (returns a numpy array)
      - Add a Weaviate object with vector=qv.tolist() and all 6 properties
        populated from the row.

    Use `client.batch` for efficiency. Call `client.batch.flush()` (or use
    a `with client.batch as batch:` context) so the final batch commits.

    Returns the count of ingested objects (verify via Aggregate query, or
    simply track count as you ingest).
    """
    rows = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    texts = [row["text"] for row in rows]

    vectors = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
    )

    client.batch.configure(batch_size=100)

    with client.batch as batch:
        for row, vec in zip(rows, vectors):
            props = {
                "doc_id": row["id"],
                "subset": row["subset"],
                "title": row["title"],
                "question_text": row["question_text"],
                "answer_text": row["answer_text"],
                "text": row["text"],
            }

            batch.add_data_object(
                data_object=props,
                class_name=CLASS_NAME,
                vector=vec.tolist(),
            )

    return len(rows)


def bm25_search(client: weaviate.Client, query: str, k: int) -> list[str]:
    """BM25 retrieval. Return ordered list of doc_id strings, length <= k.

    Use:
        client.query.get("Post", ["doc_id"]).with_bm25(query=query).with_limit(k).do()
    """
    result = (
        client.query
        .get(CLASS_NAME, ["doc_id"])
        .with_bm25(query=query)
        .with_limit(k)
        .do()
    )

    hits = result["data"]["Get"].get(CLASS_NAME, [])

    return [hit["doc_id"] for hit in hits]


def dense_search(client: weaviate.Client, query: str, k: int, embedder) -> list[str]:
    """Dense retrieval. Embed the query with the same embedder used at ingest.

    Use:
        client.query.get("Post", ["doc_id"]).with_near_vector({"vector": qv}).with_limit(k).do()
    """
    qv = embedder.encode(query).tolist()

    result = (
        client.query
        .get(CLASS_NAME, ["doc_id"])
        .with_near_vector({"vector": qv})
        .with_limit(k)
        .do()
    )

    hits = result["data"]["Get"].get(CLASS_NAME, [])

    return [hit["doc_id"] for hit in hits]


def hybrid_search(client: weaviate.Client, query: str, k: int, embedder, alpha: float = 0.5) -> list[str]:
    """Hybrid retrieval. alpha=0.5 is the canonical mix for the base assignment.

    Use:
        client.query.get("Post", ["doc_id"]).with_hybrid(query=query, vector=qv, alpha=alpha).with_limit(k).do()
    """
    qv = embedder.encode(query).tolist()

    result = (
        client.query
        .get(CLASS_NAME, ["doc_id"])
        .with_hybrid(
            query=query,
            vector=qv,
            alpha=alpha,
        )
        .with_limit(k)
        .do()
    )

    hits = result["data"]["Get"].get(CLASS_NAME, [])

    return [hit["doc_id"] for hit in hits]


def evaluate_retriever(eval_path: str, search_fn: Callable, k_values=(5, 10)) -> dict:
    """Evaluate a retriever against the labeled set.

    For each (query, gold_doc_id, query_type) row:
      - Call search_fn(query, k=max(k_values))  # one call per query
      - Compute hit@5 (gold in top-5) and hit@10 (gold in top-10)
      - Compute MRR contribution: 1/rank (1-indexed) if gold in top-10, else 0

    Return:
        {
          "recall@5": <mean hit@5>,
          "recall@10": <mean hit@10>,
          "mrr": <mean MRR>,
          "by_type": {  # REQUIRED — used in the comparison brief
            "factoid": {"recall@5": ..., "recall@10": ..., "mrr": ...},
            "paraphrastic": {"recall@5": ..., "recall@10": ..., "mrr": ...}
          }
        }
    """
    max_k = max(k_values)

    rows = []

    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    stats = defaultdict(lambda: {
        "count": 0,
        "hit5": 0,
        "hit10": 0,
        "mrr": 0.0,
    })

    overall = {
        "count": 0,
        "hit5": 0,
        "hit10": 0,
        "mrr": 0.0,
    }

    for row in rows:
        query = row["query"]
        gold = row["gold_doc_id"]
        qtype = row["query_type"]

        results = search_fn(query, k=max_k)

        top5 = results[:5]
        top10 = results[:10]

        hit5 = int(gold in top5)
        hit10 = int(gold in top10)

        rr = 0.0
        if gold in top10:
            rank = top10.index(gold) + 1
            rr = 1.0 / rank

        overall["count"] += 1
        overall["hit5"] += hit5
        overall["hit10"] += hit10
        overall["mrr"] += rr

        stats[qtype]["count"] += 1
        stats[qtype]["hit5"] += hit5
        stats[qtype]["hit10"] += hit10
        stats[qtype]["mrr"] += rr

    results = {
        "recall@5": overall["hit5"] / overall["count"],
        "recall@10": overall["hit10"] / overall["count"],
        "mrr": overall["mrr"] / overall["count"],
        "by_type": {},
    }

    for qtype, s in stats.items():
        results["by_type"][qtype] = {
            "recall@5": s["hit5"] / s["count"],
            "recall@10": s["hit10"] / s["count"],
            "mrr": s["mrr"] / s["count"],
        }

    return results


def main(client, embedder):
    eval_path = "data/retrieval_eval.jsonl"

    print("\n=== Evaluating BM25 ===")
    bm25_metrics = evaluate_retriever(
        eval_path,
        lambda q, k: bm25_search(client, q, k),
    )

    print("\n=== Evaluating Dense ===")
    dense_metrics = evaluate_retriever(
        eval_path,
        lambda q, k: dense_search(client, q, k, embedder),
    )

    print("\n=== Evaluating Hybrid ===")
    hybrid_metrics = evaluate_retriever(
        eval_path,
        lambda q, k: hybrid_search(client, q, k, embedder, alpha=0.5),
    )

    print("\n=== Metrics ===")
    print(json.dumps({
        "bm25": bm25_metrics,
        "dense": dense_metrics,
        "hybrid": hybrid_metrics,
    }, indent=2))

    bm25_wins = []
    dense_wins = []

    with open(eval_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    for row in rows:
        query = row["query"]
        gold = row["gold_doc_id"]

        bm25_res = bm25_search(client, query, 10)
        dense_res = dense_search(client, query, 10, embedder)

        bm25_rank = bm25_res.index(gold) + 1 if gold in bm25_res else 999
        dense_rank = dense_res.index(gold) + 1 if gold in dense_res else 999

        if bm25_rank < dense_rank:
            bm25_wins.append((query, gold, bm25_rank, dense_rank))

        elif dense_rank < bm25_rank:
            dense_wins.append((query, gold, bm25_rank, dense_rank))

    bm25_wins.sort(key=lambda x: x[3] - x[2], reverse=True)
    dense_wins.sort(key=lambda x: x[2] - x[3], reverse=True)

    print("\n=== Top BM25 Wins ===")
    for w in bm25_wins[:2]:
        print(w)

    print("\n=== Top Dense Wins ===")
    for w in dense_wins[:2]:
        print(w)


if __name__ == "__main__":
    client = weaviate.Client("http://localhost:8080")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    main(client, embedder)