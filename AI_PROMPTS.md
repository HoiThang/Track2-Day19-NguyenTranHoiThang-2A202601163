# AI Prompts for Lab Day 19 — NB1 to NB8

Generated for vibe-coding workflow. Each prompt follows the pattern:
- Specs in → Code out
- Validate before generate (for algorithms)
- Minimal repro → Expand

---

## NB1 — Embed + Upsert Loop

```markdown
# NB1: Embed Corpus + Index to Qdrant

## Context
- Corpus: 1000 Vietnamese tech docs (data/corpus_vn.jsonl)
- Each doc: {doc_id, topic, title, text}
- Embedding model: BAAI/bge-small-en-v1.5 (384-dim vectors)
- Target: Qdrant collection "lab19" with cosine distance

## Spec

### Input
- `docs`: list[dict] — loaded from corpus_vn.jsonl
- `embedder`: TextEmbedding instance
- `client`: QdrantClient (":memory:" mode)

### Output
- `client.count("lab19")` == 1000

### Code Pattern

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Setup
client = QdrantClient(":memory:")
client.create_collection(
    collection_name="lab19",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Batch embed + upsert
BATCH = 64  # sweet spot for fastembed CPU-bound
points: list[PointStruct] = []

for start in range(0, len(docs), BATCH):
    batch = docs[start:start + BATCH]
    texts = [d["title"] + " " + d["text"] for d in batch]
    vectors = list(embedder.embed(texts))  # returns iterator of np.ndarray
    
    for i, (d, v) in enumerate(zip(batch, vectors)):
        points.append(PointStruct(
            id=start + i,
            vector=v.tolist(),  # convert np.ndarray → list[float]
            payload={
                "doc_id": d["doc_id"],
                "topic": d["topic"],
                "title": d["title"],
            },
        ))

client.upsert(collection_name="lab19", points=points)
```

### Key Points
- `embedder.embed(texts)` returns an iterator — consume with `list()` or `next()`
- `v.tolist()` converts numpy array to Python list for Qdrant
- `PointStruct.id` must be int or str, unique within collection
- Payload stores doc_id + topic + title (NOT full text for memory efficiency)

## Deliverable
- Verify: `client.count("lab19").count == 1000`
- Test paraphrase query: "phương pháp tự động mở rộng hạ tầng" should return mostly "cloud" topic
```

---

## NB2 — RRF (Reciprocal Rank Fusion)

```markdown
# NB2: Reciprocal Rank Fusion (RRF)

## Context
- Two retrievers: BM25 (keyword) + Vector (semantic)
- Both return ranked lists of doc_ids
- Goal: combine rankings into one unified score

## Spec

### Input
```python
def search_hybrid(query: str, top_k: int = 10, rrf_k: int = 60) -> list[str]:
    depth = max(top_k * 5, 50)  # fetch deeper for better fusion
    kw_ids = search_keyword(query, depth)   # list[str], rank 1..depth
    sem_ids = search_semantic(query, depth)  # list[str], rank 1..depth
```

### Output
- `list[str]` of top_k doc_ids sorted by RRF score descending

### RRF Formula (CRITICAL)
```
score(d) = Σ_r 1/(k + rank_r(d))

where:
  - k = rrf_k (default: 60)
  - rank_r(d) = 1-BASED position (first = 1, NOT 0)
  - r = each retriever (keyword, semantic)
```

### Implementation

```python
def search_hybrid(query: str, top_k: int = 10, rrf_k: int = 60) -> list[str]:
    depth = max(top_k * 5, 50)
    kw_ids = search_keyword(query, depth)
    sem_ids = search_semantic(query, depth)

    # Accumulate RRF scores
    rrf_scores: dict[str, float] = {}
    
    # Keyword scores: rank starts at 1
    for rank, doc_id in enumerate(kw_ids, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    
    # Semantic scores: rank starts at 1
    for rank, doc_id in enumerate(sem_ids, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

    # Sort by score descending, return top_k
    return [doc_id for doc_id, _ in sorted(rrf_scores.items(), key=lambda kv: -kv[1])[:top_k]]
```

### Common Mistakes to Avoid
| ❌ Wrong | ✅ Correct |
|----------|-----------|
| `1/rank` (rank=0 → division by zero) | `1/(rrf_k + rank)` |
| `enumerate(ids, start=0)` (0-based) | `enumerate(ids, start=1)` (1-based) |
| `sum(1/rank for rank...)` | accumulate in dict |

### Validation
- Query "co giãn linh hoạt theo nhu cầu sử dụng"
- hybrid top-3 should be mostly "cloud" topic
- hybrid avg Precision@10 should beat both keyword AND semantic
```

---

## NB3 — FastAPI + Latency Benchmark

```markdown
# NB3: FastAPI /search Endpoint + Latency Benchmark

## Part A: API Endpoint (app/main.py)

### Spec

```python
# Response model
class SearchResponse(BaseModel):
    query: str
    mode: Literal["keyword", "semantic", "hybrid"]
    top_k: int
    latency_ms: float  # server-side, exclude network
    hits: list[SearchHitOut]

@app.get("/search")
def search(
    q: str = Query(..., min_length=1),
    mode: Literal["keyword", "semantic", "hybrid"] = Query("hybrid"),
    top_k: int = Query(10, ge=1, le=100),
    rrf_k: int = Query(60, ge=1, le=200),
) -> SearchResponse:
    t0 = time.perf_counter()
    hits = searcher.search(q, mode=mode, top_k=top_k, rrf_k=rrf_k)
    latency_ms = (time.perf_counter() - t0) * 1000
    return SearchResponse(query=q, mode=mode, top_k=top_k, latency_ms=latency_ms, hits=...)
```

### Key Points
- Use `time.perf_counter()` for server-side timing
- `latency_ms` is INTERNAL — does NOT include network
- Rubric asserts P99 < 50ms SERVER-SIDE

## Part B: Latency Benchmark

### Spec

```python
def percentile(values: list[float], p: float) -> float:
    """Calculate percentile from sorted list."""
    n = len(values)
    if n == 0:
        return 0.0
    return sorted(values)[min(int(n * p), n - 1)]

def benchmark_mode(mode: str, reps: int = 2) -> dict[str, float]:
    server_latencies: list[float] = []
    wall_latencies: list[float] = []
    
    for _ in range(reps):
        for q in golden_queries:
            t0 = time.perf_counter()
            r = httpx.get(f"{URL}/search", params={"q": q["query"], "mode": mode})
            wall_latencies.append((time.perf_counter() - t0) * 1000)
            server_latencies.append(r.json()["latency_ms"])
    
    return {
        "p50_server": percentile(server_latencies, 0.50),
        "p95_server": percentile(server_latencies, 0.95),
        "p99_server": percentile(server_latencies, 0.99),
        "p99_wall":   percentile(wall_latencies, 0.99),
    }
```

### Rubric Check
```python
hybrid_p99 = results["hybrid"]["p99_server"]
assert hybrid_p99 < 50, f"Hybrid P99 {hybrid_p99}ms must be < 50ms"
```
```

---

## NB4 — Feast Feature Store

```markdown
# NB4: Feast Feature Store — 3 Feature Views + Online Lookup

## Part A: Generate Parquet Data

```python
from datetime import datetime, timedelta, timezone
import polars as pl

NOW = datetime.now(timezone.utc).replace(microsecond=0)

def make_user_profile(n_users: int = 100) -> pl.DataFrame:
    return pl.DataFrame({
        "user_id": [f"u_{i:03d}" for i in range(n_users)],
        "reading_speed_wpm": [180 + (i * 7) % 200 for i in range(n_users)],
        "preferred_language": ["vi" if i % 3 != 0 else "en" for i in range(n_users)],
        "topic_affinity": ["ai_ml", "cloud", "security", "database", "devops"][i % 5],
        "event_timestamp": [NOW - timedelta(hours=i % 48) for i in range(n_users)],
    })

def make_item_popularity(n_items: int = 1000) -> pl.DataFrame:
    return pl.DataFrame({
        "doc_id": [f"item_{i:04d}" for i in range(n_items)],
        "click_count_24h": [(i * 13) % 500 for i in range(n_items)],
        "ctr_7d": [round(((i * 7) % 100) / 100.0, 3) for i in range(n_items)],
        "avg_dwell_seconds": [10.0 + (i * 0.7) % 90 for i in range(n_items)],
        "event_timestamp": [NOW - timedelta(minutes=i % 720) for i in range(n_items)],
    })

def make_query_velocity(n_users: int = 100) -> pl.DataFrame:
    return pl.DataFrame({
        "user_id": [f"u_{i:03d}" for i in range(n_users)],
        "queries_last_hour": [(i * 11) % 50 for i in range(n_users)],
        "distinct_topics_24h": [1 + (i * 3) % 10 for i in range(n_users)],
        "event_timestamp": [NOW - timedelta(minutes=i % 30) for i in range(n_users)],
    })

make_user_profile().write_parquet("app/feast_repo/data/user_profile.parquet")
make_item_popularity().write_parquet("app/feast_repo/data/item_popularity.parquet")
make_query_velocity().write_parquet("app/feast_repo/data/query_velocity.parquet")
```

## Part B: CLI Commands

```bash
cd app/feast_repo
feast apply
feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
```

## Part C: Online Lookup + P99

```python
from feast import FeatureStore

fs = FeatureStore(repo_path="app/feast_repo")

REQUEST_FEATURES = [
    "user_profile_features:reading_speed_wpm",
    "user_profile_features:preferred_language",
    "user_profile_features:topic_affinity",
    "query_velocity_features:queries_last_hour",
    "query_velocity_features:distinct_topics_24h",
]

latencies: list[float] = []
for i in range(100):
    user_id = f"u_{i:03d}"
    t0 = time.perf_counter()
    fs.get_online_features(features=REQUEST_FEATURES, entity_rows=[{"user_id": user_id}]).to_dict()
    latencies.append((time.perf_counter() - t0) * 1000)

latencies.sort()
print(f"P50={latencies[50]:.2f}ms P95={latencies[95]:.2f}ms P99={latencies[99]:.2f}ms")
```
```

---

## NB5 — Filtered ANN Strategies

```markdown
# NB5: Filtered Vector Search — 3 Strategies Compared

## Context
- 1000 docs with metadata: tenant, access, topic, published_ts
- Query: "tự động mở rộng hệ thống theo lưu lượng"
- Filter example: tenant="acme" AND published_ts >= 20260101

## Three Strategies

### 1. post_filter (NAIVE — has recall cliff)

```python
def post_filter(query: str, predicate, k: int = 10, fetch_k: int | None = None) -> list[str]:
    fetch_k = fetch_k or k
    qv = embed(query)
    hits = qdrant.query_points(collection_name="lab19_filtered", query=qv.tolist(), limit=fetch_k)
    return [h.payload["doc_id"] for h in hits if predicate(h.payload)][:k]
```

### 2. pre_filter (CORRECT but slow)

```python
def pre_filter(query: str, predicate, k: int = 10) -> list[str]:
    qv = embed(query)
    matching_indices = [i for i, d in enumerate(docs) if predicate(d)]
    sub_vectors = vectors[matching_indices]
    sims = (sub_vectors @ qv) / (norm(sub_vectors) * norm(qv) + 1e-12)
    order = argsort(-sims)[:k]
    return [docs[matching_indices[i]]["doc_id"] for i in order]
```

### 3. filtered_ANN (PRODUCTION)

```python
def filtered_ann(query: str, qfilter: Filter, k: int = 10) -> list[str]:
    qv = embed(query)
    hits = qdrant.query_points(
        collection_name="lab19_filtered",
        query=qv.tolist(),
        query_filter=qfilter,  # <-- filter INSIDE the index
        limit=k,
    )
    return [h.payload["doc_id"] for h in hits]
```

## Ground Truth (CRITICAL)

```python
def exact_top_k(qv, predicate, k) -> list[str]:
    """Brute-force cosine over MATCHING subset only.
    THIS is ground truth, NOT top-K without filter."""
    matching_indices = [i for i, d in enumerate(docs) if predicate(d)]
    sub_vectors = vectors[matching_indices]
    sims = (sub_vectors @ qv) / (norm(sub_vectors) * norm(qv) + 1e-12)
    order = argsort(-sims)[:k]
    return [docs[matching_indices[i]]["doc_id"] for i in order]
```
```

---

## NB6 — Agentic Retrieval

```markdown
# NB6: Retrieval as a Tool + Rule-Based Planner

## Tool Definition

```python
SEARCH_TOOL = {
    "name": "search_docs",
    "description": "Search Vietnamese technical docs corpus. One question per call.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "topic": {"type": "string", "enum": ["cloud", "ai_ml", "security", ...]},
            "top_k": {"type": "integer", "default": 8},
        },
        "required": ["query"],
    },
}
```

## Rule-Based Planner

```python
SPLIT_RE = re.compile(r"\s+(?:và|hoặc|so với)\s+", re.IGNORECASE)

TOPIC_HINTS = {
    "cloud": ["cloud", "đám mây", "kubernetes", "serverless"],
    "ai_ml": ["ai", "mô hình", "embedding", "llm"],
    # ...
}

class RuleBasedPlanner:
    def __init__(self, budget: int = 16, use_filters: bool = True):
        self.budget = budget
        self.use_filters = use_filters
    
    def detect_topic(self, text: str) -> str | None:
        low = text.lower()
        best, best_n = None, 0
        for topic, hints in TOPIC_HINTS.items():
            n = sum(1 for h in hints if h in low)
            if n > best_n:
                best, best_n = topic, n
        return best
    
    def plan(self, question: str) -> list[ToolArgs]:
        parts = [p.strip() for p in SPLIT_RE.split(question) if p and len(p.strip()) > 3]
        if len(parts) < 2:
            parts = [question]
        per = max(1, self.budget // len(parts))
        return [ToolArgs(query=p, topic=self.detect_topic(p) if self.use_filters else None, top_k=per) for p in parts]
```

## Agent with Reflection

```python
class Agent:
    def __init__(self, tool, planner, min_evidence: int = 4):
        self.tool = tool
        self.planner = planner
        self.min_evidence = min_evidence
    
    def answer(self, question: str) -> list[str]:
        seen = []
        for args in self.planner.plan(question):
            call = self.tool(args)
            if len(call.doc_ids) < self.min_evidence and (args.topic):
                call = self.tool(ToolArgs(query=args.query, top_k=args.top_k))
            for d in call.doc_ids:
                if d not in seen:
                    seen.append(d)
        return seen
```

## IMPORTANT: Same Budget for Comparison
- `budget=16` MUST be identical for single-shot and agentic
- Agent should NOT get more documents than single-shot
```

---

## NB7 — Semantic Cache

```markdown
# NB7: Semantic Cache with Security

## Cache Implementation

```python
class SemanticCache:
    def __init__(self, client, embedder, threshold=0.75, ttl_s=3600.0, namespaced=True):
        self.threshold = threshold
        self.ttl_s = ttl_s
        self.namespaced = namespaced
        self._clock = 0.0
        self._next_id = 0
    
    def get(self, tenant: str, question: str) -> CacheHit | None:
        qf = None
        if self.namespaced:
            qf = Filter(must=[FieldCondition(key="tenant", match=MatchValue(value=tenant))])
        
        pts = self.client.query_points(collection_name="cache", query=self._embed(question), query_filter=qf, limit=1).points
        
        if not pts or pts[0].score < self.threshold:
            return None
        
        p = pts[0]
        age = self._clock - p.payload["ts"]
        if self.ttl_s and age > self.ttl_s:
            self.client.delete(collection_name="cache", points=[p.id])
            return None
        
        return CacheHit(answer=p.payload["answer"], question=p.payload["question"], score=float(p.score), tenant=p.payload["tenant"], age_s=age)
    
    def put(self, tenant: str, question: str, answer: str) -> None:
        self.client.upsert(collection_name="cache", points=[PointStruct(id=self._next_id, vector=self._embed(question), payload={"tenant": tenant, "question": question, "answer": answer, "ts": self._clock})])
        self._next_id += 1
```

## Threshold Sweep (TWO columns!)

```python
def sweep_thresholds(cache, warm_queries, cold_queries):
    """Measure BOTH: 'saved' (good hits) AND 'wrong' (bad hits)."""
    for q in warm_queries:
        cache.put("acme", q["query"], f"ANSWER::{q['query_id']}")
    
    for th in (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        saved = sum(1 for q in warm_queries if cache.peek("acme", q["query"]) and cache.peek("acme", q["query"])[0] >= th) / len(warm_queries)
        wrong = sum(1 for q in cold_queries if cache.peek("acme", q["query"]) and cache.peek("acme", q["query"])[0] >= th) / len(cold_queries)
        print(f"th={th:.2f} saved={saved:.0%} wrong={wrong:.0%}")
```

## Tenant Isolation Demo

```python
leaky = SemanticCache(client, embedder, namespaced=False)
leaky.put("acme", "revenue Q3", "ACME: 4.2B VND")
stolen = leaky.get("globex", "revenue Q3")  # SECURITY INCIDENT!

safe = SemanticCache(client, embedder, namespaced=True)
safe.put("acme", "revenue Q3", "ACME: 4.2B VND")
blocked = safe.get("globex", "revenue Q3")  # Correct: None
```
```

---

## NB8 — Feature Engineering + Leakage

```markdown
# NB8: Feature Engineering + Leakage Detection

## Six Feature Families

```python
def window_aggregates(events: pd.DataFrame, windows=("1h", "24h", "7d")) -> pd.DataFrame:
    df = events.sort_values("event_timestamp").copy()
    out = df[["user_id", "event_timestamp", "topic", "query_len", "clicked"]].copy()
    
    for w in windows:
        col = f"searches_{w}"
        vals = []
        delta = np.timedelta64(window_to_seconds(w), "s")
        for user, g in df.groupby("user_id", sort=False):
            ts = g["event_timestamp"].values.astype("datetime64[s]")
            cnt = [int((ts[:i] > ts[i] - delta).sum()) for i in range(len(ts))]
            vals.append(pd.Series(cnt, index=g.index))
        out[col] = pd.concat(vals).sort_index()
    
    # Ratio vs user's own expanding mean (shifted = causal)
    out["query_len_vs_user_avg"] = df["query_len"] / df.groupby("user_id")["query_len"].transform(lambda s: s.shift().expanding().mean())
    out["prev_query_len"] = df.groupby("user_id")["query_len"].shift()
    out["query_len_delta"] = df["query_len"] - out["prev_query_len"]
    prev_ts = df.groupby("user_id")["event_timestamp"].shift()
    out["seconds_since_last"] = (df["event_timestamp"] - prev_ts).dt.total_seconds()
    return out
```

## Target Encoding Leakage

```python
def leakage_experiment(df: pd.DataFrame, col: str, target: str = "clicked"):
    """CORRECT: split FIRST, encode on train only."""
    rng = np.random.default_rng(7)
    is_test = rng.random(len(df)) < 0.3
    train, test = df.loc[~is_test], df.loc[is_test]
    prior = train[target].mean()
    
    # Frequency (safe)
    freq = train[col].value_counts(normalize=True)
    
    # Naive target (leaky)
    means = train.groupby(col)[target].mean()
    
    # In-fold (correct)
    infold = target_encode_in_fold(train, col, target)
    
    return pd.DataFrame({
        "encoding": ["frequency", "target-naive", "target-in-fold"],
        "train_auc": [auc(train[col].map(freq), train[target]), auc(train[col].map(means), train[target]), auc(infold, train[target])],
        "test_auc": [auc(test[col].map(freq).fillna(0.0), test[target]), auc(test[col].map(means).fillna(prior), test[target]), auc(test[col].map(means).fillna(prior), test[target])],
    }).assign(gap=lambda x: x["train_auc"] - x["test_auc"])
```

## PIT vs Latest Join

```python
def latest_join(entity_df, feature_df):
    """WRONG: uses future values."""
    latest = feature_df.sort_values("event_timestamp").groupby("user_id").tail(1)[["user_id", "feature_value"]]
    return entity_df.merge(latest, on="user_id", how="left")

def pit_join(entity_df, feature_df):
    """CORRECT: as-of join."""
    e = entity_df.sort_values("event_timestamp")
    f = feature_df.sort_values("event_timestamp")
    return pd.merge_asof(e, f[["user_id", "event_timestamp", "feature_value"]], on="event_timestamp", by="user_id", direction="backward")
```

## On-Demand Feature (Feast)

```python
@on_demand_feature_view(sources=[user_spend_stats, txn_request], schema=[Field(name="amount_vs_avg", dtype=Float64), Field(name="is_spike", dtype=Int64)], mode="python")
def amount_vs_avg(inputs: dict) -> dict:
    ratios = [(a / m) if m else 0.0 for a, m in zip(inputs["amount"], inputs["avg_amount_7d"])]
    return {"amount_vs_avg": ratios, "is_spike": [int(r > 3.0) for r in ratios]}
```

## KEY: Order Matters
- Split FIRST → encode on train only → measure both
- If encode BEFORE split, holdout labels are baked in
- Gap > 0.30 on session_id = massive leakage
```

---

## Think-Hard Summary

| Notebook | Component | Why Think Hard? |
|----------|-----------|----------------|
| NB1 | Embedding model | bge-small-en vs bge-m3 cho tiếng Việt |
| NB2 | RRF formula | rank 1-based, k=60, not 0-based |
| NB3 | What to measure | server-side vs wall-clock, P99 not mean |
| NB4 | TTL choices | user=30d vs query_velocity=1h |
| NB5 | Ground truth | exact top-K in subset, not corpus |
| NB6 | Same budget | agentic must NOT get more docs |
| NB7 | Two columns | "saved" AND "wrong" hit rate |
| NB8 | Split before encode | order matters for leakage |
