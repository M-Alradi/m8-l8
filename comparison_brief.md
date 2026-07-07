# Comparison Brief — Module 8 Lab

## Metrics Table

| Retriever | recall@5 | recall@10 | MRR | factoid recall@5 | paraphrastic recall@5 |
|---|---|---|---|---|---|
| BM25 | 0.550 | 0.650 | 0.536 | 1.000 | 0.100 |
| Dense | 0.900 | 0.933 | 0.670 | 0.833 | 0.967 |
| Hybrid (α=0.5) | 0.833 | 0.983 | 0.698 | 0.967 | 0.700 |

---

## Where BM25 Wins

BM25 performs best on queries where exact keywords or rare identifiers appear.

1. **"What is the AskDeveloperIfThereIsDocumentationLyingSomewhereAround pseudocode about?"**  
   - Gold: `programmers:121844`  
   - BM25 rank = 1, Dense rank = not retrieved (999)  
   - BM25 wins because the query contains a very rare, exact technical phrase that matches tokens in the document.

2. **"What is the FollowSymLinks RewriteEngine .htaccess pattern for redirecting yoursite.com?"**  
   - Gold: `webmasters:20302`  
   - BM25 rank = 1, Dense rank = not retrieved (999)  
   - BM25 succeeds due to exact overlap of configuration keywords like `.htaccess`, `RewriteEngine`, and `FollowSymLinks`.

---

## Where Dense Wins

Dense retrieval wins when meaning is preserved but wording differs.

1. **"Does having a dynamic URL for all pages in my blog put me at disadvantage SEO wise?"**  
   - Gold: `webmasters:20914`  
   - Dense rank = 1, BM25 rank = not retrieved (999)  
   - Dense works because it understands semantic meaning around SEO impact, not exact keywords.

2. **"How to make my page load faster?"**  
   - Gold: `webmasters:15380`  
   - Dense rank = 1, BM25 rank = not retrieved (999)  
   - Dense succeeds due to semantic similarity with performance optimization concepts.

---

## Alpha Recommendation

I recommend **α = 0.5** for this corpus.

The results show a clear split:
- BM25 is very strong on exact technical terms and identifiers (factoid queries).
- Dense retrieval is significantly better on paraphrased or conceptual queries.

A balanced hybrid (α=0.5) is justified because it:
- Preserves BM25’s precision for structured technical tokens
- Retains dense retrieval’s strength on semantic understanding
- Produces the best overall balance in recall@10 and MRR

Slightly favoring BM25 (α in the 0.5–0.6 range) is reasonable given its perfect factoid recall.

---

## Schema Choice — Cosine vs Dot Product

The schema uses:

```json
{"distance": "cosine"}