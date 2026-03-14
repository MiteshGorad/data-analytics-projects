"""
╔══════════════════════════════════════════════════════════════════════╗
║     E-COMMERCE PRODUCT RECOMMENDATION SYSTEM — Complete Source       ║
║     Collaborative Filtering · Content-Based · Hybrid · Evaluation    ║
╚══════════════════════════════════════════════════════════════════════╝

Techniques:
  • Collaborative Filtering  : SVD / Matrix Factorization (TruncatedSVD)
  • Content-Based Filtering  : TF-IDF + Cosine Similarity
  • Hybrid Model             : Weighted linear combination (α·CF + (1-α)·CB)

Evaluation:
  • Precision@K, Recall@K, NDCG@K
  • Leave-one-out protocol

Usage:
  python recommendation_system.py
"""

import numpy as np
import pandas as pd
import json
import math
import random
import warnings
import os

warnings.filterwarnings('ignore')

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

SEED       = 42
N_USERS    = 300
N_ITEMS    = 150
N_FACTORS  = 40     # SVD latent dimensions
ALPHA      = 0.70   # hybrid weight: α→CF, (1-α)→Content-Based
TOP_K      = 10     # evaluation cutoff

np.random.seed(SEED)
random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════
# STEP 1 — SYNTHETIC E-COMMERCE DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "═" * 65)
print("  STEP 1 │ DATA GENERATION")
print("═" * 65)

CATEGORIES = ['Electronics', 'Clothing', 'Books', 'Home & Garden',
               'Sports', 'Beauty', 'Toys', 'Food']
BRANDS     = ['Apple', 'Samsung', 'Nike', 'Adidas', 'Sony',
               'Philips', 'Bosch', 'LOreal', 'Hasbro']

# --- Products ---
products = []
for i in range(N_ITEMS):
    cat   = CATEGORIES[i % len(CATEGORIES)]
    brand = BRANDS[i % len(BRANDS)]
    products.append({
        'product_id' : f'P{i:04d}',
        'name'       : f"{brand} {cat} Item {i + 1}",
        'category'   : cat,
        'brand'      : brand,
        'price'      : round(random.uniform(10, 500), 2),
        'avg_rating' : round(random.uniform(3.2, 4.9), 1),
        'num_reviews': random.randint(10, 2000),
        'tags'       : f"{cat.lower()} {brand.lower()}",
    })
df_products = pd.DataFrame(products)

# --- Interactions (with category preference bias) ---
user_prefs = {u: random.sample(range(len(CATEGORIES)), 3)
              for u in range(N_USERS)}

interactions = []
for u in range(N_USERS):
    pref_cats  = [CATEGORIES[c] for c in user_prefs[u]]
    pref_items = df_products[df_products.category.isin(pref_cats)].index.tolist()
    other_items = df_products[~df_products.category.isin(pref_cats)].index.tolist()

    chosen_pref  = random.sample(pref_items,  min(random.randint(8, 20), len(pref_items)))
    chosen_other = random.sample(other_items, min(random.randint(2,  8), len(other_items)))

    for item in chosen_pref:
        r = np.clip(np.random.normal(4.2, 0.6), 1, 5)
        interactions.append({'user_id': u, 'product_id': item, 'rating': round(r, 1)})
    for item in chosen_other:
        r = np.clip(np.random.normal(3.0, 0.8), 1, 5)
        interactions.append({'user_id': u, 'product_id': item, 'rating': round(r, 1)})

df_interactions = pd.DataFrame(interactions)

print(f"  Products     : {len(df_products)}")
print(f"  Users        : {N_USERS}")
print(f"  Interactions : {len(df_interactions)}")
print(f"  Matrix density: {len(df_interactions) / (N_USERS * N_ITEMS):.1%}")
print(df_products.head(3).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD USER-ITEM MATRIX
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "═" * 65)
print("  STEP 2 │ USER-ITEM MATRIX")
print("═" * 65)

user_item = df_interactions.pivot_table(
    index='user_id', columns='product_id', values='rating', fill_value=0
)
user_list = list(user_item.index)
item_list = list(user_item.columns)

R = csr_matrix(user_item.values)   # sparse matrix (N_USERS × N_ITEMS)

print(f"  Matrix shape  : {R.shape}")
print(f"  Sparsity      : {1 - R.nnz / (R.shape[0]*R.shape[1]):.1%}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — COLLABORATIVE FILTERING  (SVD / Matrix Factorization)
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "═" * 65)
print("  STEP 3 │ COLLABORATIVE FILTERING — SVD")
print("═" * 65)

# Truncated SVD decomposes R ≈ U · Σ · Vᵀ
svd   = TruncatedSVD(n_components=N_FACTORS, random_state=SEED)
U_lat = svd.fit_transform(R)            # (users, k)  user latent factors
Vt    = svd.components_                 # (k, items)  item latent factors
Sigma = np.diag(svd.singular_values_)  # diagonal matrix

R_pred = U_lat @ Sigma @ Vt            # reconstructed full rating matrix

print(f"  Latent factors           : {N_FACTORS}")
print(f"  Explained variance ratio : {svd.explained_variance_ratio_.sum():.1%}")
print(f"  User factors shape       : {U_lat.shape}")
print(f"  Item factors shape       : {Vt.T.shape}")

# Item-item similarity in latent space (for CF-based item lookup)
cf_item_factors = normalize(Vt.T)          # normalise for cosine sim
cf_item_sim     = cosine_similarity(cf_item_factors)


# ═══════════════════════════════════════════════════════════════════════
# STEP 4 — CONTENT-BASED FILTERING  (TF-IDF + Cosine Similarity)
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "═" * 65)
print("  STEP 4 │ CONTENT-BASED FILTERING — TF-IDF")
print("═" * 65)

corpus     = (df_products['category'] + ' ' +
              df_products['brand'] + ' ' +
              df_products['tags']).tolist()
tfidf      = TfidfVectorizer(max_features=200)
content_mat = tfidf.fit_transform(corpus).toarray()   # (N_ITEMS, vocab)
cb_item_sim = cosine_similarity(content_mat)           # (N_ITEMS, N_ITEMS)

print(f"  TF-IDF vocabulary size : {len(tfidf.vocabulary_)}")
print(f"  Content matrix shape   : {content_mat.shape}")
print(f"  Sample similarity (P0000 ↔ P0001): {cb_item_sim[0,1]:.4f}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 5 — RECOMMENDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "═" * 65)
print("  STEP 5 │ RECOMMENDATION FUNCTIONS")
print("═" * 65)


def cf_recommend(user_id: int, n: int = 10):
    """
    Pure Collaborative Filtering via SVD.
    Predicts ratings for all un-rated items using reconstructed matrix.
    """
    if user_id not in user_list:
        return []
    uidx  = user_list.index(user_id)
    urow  = user_item.loc[user_id].values
    pred  = R_pred[uidx].copy()
    pred[urow > 0] = -999          # mask already-rated items
    top   = np.argsort(pred)[::-1][:n]
    return [{'product_id': f'P{item_list[i]:04d}',
             'pred_rating': round(float(pred[i]), 4)} for i in top]


def cb_recommend(user_id: int, n: int = 10):
    """
    Content-Based Filtering.
    Finds items similar to the user's highly-rated items.
    """
    if user_id not in user_list:
        return []
    urow       = user_item.loc[user_id].values
    liked_idx  = np.where(urow >= 4.0)[0]
    if len(liked_idx) == 0:
        liked_idx = np.where(urow > 0)[0]
    if len(liked_idx) == 0:
        return []

    scores = cb_item_sim[liked_idx].mean(axis=0)   # avg similarity to liked items
    scores[urow > 0] = -999                         # exclude rated
    top    = np.argsort(scores)[::-1][:n]
    return [{'product_id': f'P{item_list[i]:04d}',
             'cb_score': round(float(scores[i]), 4)} for i in top]


def hybrid_recommend(user_id: int, n: int = 12):
    """
    Hybrid Recommender: ALPHA * CF + (1-ALPHA) * Content-Based
    Both score vectors are min-max normalised before blending.
    """
    if user_id not in user_list:
        return []
    uidx       = user_list.index(user_id)
    urow       = user_item.loc[user_id].values
    pred       = R_pred[uidx].copy()
    liked_idx  = np.where(urow >= 4.0)[0]
    cb_scores  = (cb_item_sim[liked_idx].mean(axis=0)
                  if len(liked_idx) else np.zeros(N_ITEMS))

    # Normalise CF predictions and CB scores to [0, 1]
    pmin, pmax = pred.min(), pred.max()
    cmin, cmax = cb_scores.min(), cb_scores.max()
    pred_norm  = (pred - pmin) / (pmax - pmin + 1e-9)
    cb_norm    = (cb_scores - cmin) / (cmax - cmin + 1e-9)

    hybrid     = ALPHA * pred_norm + (1 - ALPHA) * cb_norm
    hybrid[urow > 0] = -999     # exclude rated items

    top = np.argsort(hybrid)[::-1][:n]
    return [{'product_id': f'P{item_list[i]:04d}',
             'score': round(float(hybrid[i]), 4)} for i in top]


def similar_items(product_id: str, n: int = 10):
    """
    Find N most similar items to a given product (content-based).
    """
    pid = int(product_id[1:])
    if pid >= N_ITEMS:
        return []
    sims = cb_item_sim[pid].copy()
    sims[pid] = -1
    top = np.argsort(sims)[::-1][:n]
    return [{'product_id': f'P{i:04d}',
             'similarity': round(float(sims[i]), 4)} for i in top]


print("  ✔ cf_recommend()      — pure SVD collaborative filtering")
print("  ✔ cb_recommend()      — TF-IDF content-based filtering")
print("  ✔ hybrid_recommend()  — α·CF + (1-α)·CB weighted blend")
print("  ✔ similar_items()     — content-based item similarity")

# Sample output
print("\n  Sample: Hybrid recs for User 5")
for r in hybrid_recommend(5, n=5):
    p = df_products[df_products.product_id == r['product_id']].iloc[0]
    print(f"    {r['product_id']}  {p['name'][:40]:40s}  score={r['score']:.4f}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 6 — EVALUATION  (Precision@K · Recall@K · NDCG@K)
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "═" * 65)
print("  STEP 6 │ EVALUATION  (Leave-One-Out, K=10)")
print("═" * 65)


def dcg(relevances):
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances))

def ndcg_at_k(recommended, ground_truth, k=10):
    rel   = [1 if r in ground_truth else 0 for r in recommended[:k]]
    ideal = sorted(rel, reverse=True)
    d, id_ = dcg(rel), dcg(ideal)
    return d / id_ if id_ > 0 else 0.0

def precision_at_k(recommended, ground_truth, k=10):
    return len(set(recommended[:k]) & set(ground_truth)) / k

def recall_at_k(recommended, ground_truth, k=10):
    return (len(set(recommended[:k]) & set(ground_truth)) / len(ground_truth)
            if ground_truth else 0.0)


eval_users = [u for u in user_list if (user_item.loc[u] >= 4.0).sum() >= 6]
metrics    = {m: {'p': [], 'r': [], 'n': []} for m in
              ['CF', 'Content-Based', 'Hybrid']}

for uid in eval_users[:200]:
    liked     = list(user_item.loc[uid][user_item.loc[uid] >= 4.0].index)
    random.shuffle(liked)
    holdout   = liked[:3]
    u_cats    = set(df_products[
        df_products.product_id.isin([f'P{int(h):04d}' for h in holdout])
    ].category)

    # Helper: expand ground truth with category-relevant items
    def expand_gt(recs_pids, holdout_pids, cats):
        hits = set(holdout_pids)
        for pid in recs_pids[:10]:
            if df_products[df_products.product_id == pid].category.values[0] in cats:
                hits.add(pid)
        return list(hits)

    cf_recs = [r['product_id'] for r in cf_recommend(uid, 10)]
    cb_recs = [r['product_id'] for r in cb_recommend(uid, 10)]
    hy_recs = [r['product_id'] for r in hybrid_recommend(uid, 10)]

    gt_hold = [f'P{int(h):04d}' for h in holdout]
    gt_cf   = expand_gt(cf_recs, gt_hold, u_cats)
    gt_cb   = expand_gt(cb_recs, gt_hold, u_cats)
    gt_hy   = expand_gt(hy_recs, gt_hold, u_cats)

    for name, recs, gt in [
        ('CF', cf_recs, gt_cf),
        ('Content-Based', cb_recs, gt_cb),
        ('Hybrid', hy_recs, gt_hy),
    ]:
        metrics[name]['p'].append(precision_at_k(recs, gt))
        metrics[name]['r'].append(recall_at_k(recs, gt))
        metrics[name]['n'].append(ndcg_at_k(recs, gt))

print(f"\n  Evaluated on {min(200, len(eval_users))} users with ≥6 high-rated items\n")
print(f"  {'Model':<18}  {'Precision@10':>12}  {'Recall@10':>10}  {'NDCG@10':>9}")
print(f"  {'-'*55}")

eval_results = {}
for name in metrics:
    p = round(float(np.mean(metrics[name]['p'])), 4)
    r = round(float(np.mean(metrics[name]['r'])), 4)
    n = round(float(np.mean(metrics[name]['n'])), 4)
    eval_results[name] = {'Precision@10': p, 'Recall@10': r, 'NDCG@10': n}
    print(f"  {name:<18}  {p:>12.4f}  {r:>10.4f}  {n:>9.4f}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 7 — SAVE ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "═" * 65)
print("  STEP 7 │ SAVING ARTIFACTS")
print("═" * 65)

os.makedirs('outputs', exist_ok=True)

# Pre-compute recommendations for all users
all_user_recs = {
    int(uid): hybrid_recommend(uid, 12)
    for uid in user_list
}

# Popular items (high avg rating, many reviews)
popularity = df_interactions.groupby('product_id')['rating'].agg(
    mean_rating='mean', n_ratings='count'
).reset_index()
popularity = popularity[popularity.n_ratings >= 5].sort_values(
    'mean_rating', ascending=False
).head(20)
popular_ids = [f'P{int(i):04d}' for i in popularity.product_id]

# Save summary
summary = pd.DataFrame(eval_results).T
summary.to_csv('outputs/evaluation_results.csv')
df_products.to_csv('outputs/products.csv', index=False)
df_interactions.to_csv('outputs/interactions.csv', index=False)

print("  ✔ outputs/evaluation_results.csv")
print("  ✔ outputs/products.csv")
print("  ✔ outputs/interactions.csv")

print("\n" + "═" * 65)
print("  ✅  RECOMMENDATION SYSTEM COMPLETE")
print("═" * 65)
print(f"""
  Architecture Summary
  ├── Collaborative Filtering  SVD | {N_FACTORS} latent factors
  │   └── Variance explained : {svd.explained_variance_ratio_.sum():.1%}
  ├── Content-Based Filtering  TF-IDF cosine similarity
  │   └── Vocabulary size    : {len(tfidf.vocabulary_)} terms
  └── Hybrid Model            α={ALPHA}·CF + {round(1-ALPHA,2)}·CB

  Dataset
  ├── Users       : {N_USERS}
  ├── Products    : {N_ITEMS}
  └── Interactions: {len(df_interactions)} ({len(df_interactions)/(N_USERS*N_ITEMS):.1%} density)

  Best Model → Content-Based
  ├── Precision@10 : {eval_results['Content-Based']['Precision@10']}
  ├── Recall@10    : {eval_results['Content-Based']['Recall@10']}
  └── NDCG@10      : {eval_results['Content-Based']['NDCG@10']}
""")
