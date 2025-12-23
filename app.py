import joblib
import numpy as np
from fastapi import FastAPI, Query
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Course Recommender API")

# Load trained artifacts
bundle = joblib.load("artifacts/recommender.joblib")

user2idx = bundle["user2idx"]
idx2item = bundle["idx2item"]
U = bundle["U"]
V = bundle["V"]
courses = bundle["courses"]

course_id_to_row = bundle["course_id_to_row"]
X_course = bundle["X_course"]

def recommend_cf(user_id: int, k: int = 10):
    if user_id not in user2idx:
        return []
    uvec = U[user2idx[user_id]]
    scores = uvec @ V.T
    ranked = np.argsort(-scores)

    recs = []
    for j in ranked[:k * 10]:
        cid = int(idx2item[j])
        recs.append((cid, float(scores[j])))
        if len(recs) >= k:
            break
    return recs

def recommend_content_like(course_id: int, k: int = 10):
    if course_id not in course_id_to_row:
        return []
    i = course_id_to_row[course_id]
    sims = cosine_similarity(X_course[i], X_course).ravel()
    ranked = np.argsort(-sims)

    recs = []
    for j in ranked:
        cid = int(courses.loc[j, "course_id"])
        if cid == course_id:
            continue
        recs.append((cid, float(sims[j])))
        if len(recs) >= k:
            break
    return recs

def recommend_popular(k: int = 10):
    top = courses.sort_values("enrollment_numbers", ascending=False).head(k)
    return [(int(r.course_id), float(r.enrollment_numbers)) for _, r in top.iterrows()]

def recommend_hybrid(user_id: int, k: int = 10, alpha: float = 0.75):
    cf = recommend_cf(user_id, k=200)
    if not cf:
        return recommend_popular(k)

    # Simple and fast API hybrid:
    # use top CF item as content seed and blend
    seed_course = cf[0][0]
    content = dict(recommend_content_like(seed_course, k=200))

    combined = []
    for cid, s in cf:
        cs = float(content.get(cid, 0.0))
        combined.append((cid, alpha * s + (1 - alpha) * cs))

    combined.sort(key=lambda x: -x[1])
    return combined[:k]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/recommend")
def recommend(user_id: int = Query(...), k: int = Query(10), alpha: float = Query(0.75)):
    recs = recommend_hybrid(user_id=user_id, k=k, alpha=alpha)

    # map meta for nice response
    course_map = courses.set_index("course_id")[["course_name", "instructor", "difficulty_level", "course_price"]].to_dict("index")

    out = []
    for cid, score in recs:
        meta = course_map.get(cid, {})
        out.append({
            "course_id": cid,
            "score": score,
            **meta
        })
    return {"user_id": user_id, "k": k, "alpha": alpha, "recommendations": out}