"""
Workit - law_kb Qdrant upsert 스크립트 (Hybrid: Dense + Sparse)
파일명: yoonha_law_upsert.py
위치:   Workit/rag/yoonha_law_upsert.py

데이터:
  data/export/chunks.json          → payload
  data/export/vectors.npz          → dense 벡터 (N, 1024) float32
  data/export/sparse_weights.json  → BGE-M3 sparse lexical weights

실행:
  python rag/yoonha_law_upsert.py
"""

from __future__ import annotations

import json
import re
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
QDRANT_HOST  = "localhost"
QDRANT_PORT  = 6333
COLLECTION   = "law_kb"
VECTOR_DIM   = 1024
BATCH_SIZE   = 64
EMBED_MODEL  = "BAAI/bge-m3"

_THIS_DIR    = Path(__file__).resolve().parent
_DATA_DIR    = _THIS_DIR.parent / "data" / "export"

CHUNKS_PATH  = _DATA_DIR / "chunks.json"
VECTORS_PATH = _DATA_DIR / "vectors.npz"
SPARSE_PATH  = _DATA_DIR / "sparse_weights.json"


# ──────────────────────────────────────────
# sparse weight 변환
# ──────────────────────────────────────────
def to_sparse_vector(lexical_weights: dict, tokenizer) -> SparseVector:
    """
    BGE-M3 lexical_weights {token_str: weight} →
    Qdrant SparseVector {indices: [...], values: [...]}

    - token_str → token_id: 모델 vocab 기반 변환 (retrieval과 동일 방식)
    - 동일 token_id 중복 시 weight 합산 (Qdrant indices unique 조건 충족)
    """
    id_to_weight: dict[int, float] = {}
    for token_str, weight in lexical_weights.items():
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if isinstance(token_id, int):
            id_to_weight[token_id] = id_to_weight.get(token_id, 0.0) + float(weight)

    return SparseVector(
        indices=list(id_to_weight.keys()),
        values=list(id_to_weight.values()),
    )


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────
def main() -> None:
    print("=" * 55)
    print("Workit law_kb — Qdrant Hybrid Upsert")
    print("=" * 55)

    # ── 청크 로드 ─────────────────────────
    print(f"\n📂 chunks.json 로드: {CHUNKS_PATH}")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks: list[dict] = json.load(f)

    # chunk_id 기준 중복 제거
    chunk_map: dict[str, dict] = {}
    for c in chunks:
        chunk_map[c["chunk_id"]] = c
    print(f"   원본 {len(chunks)}개 → chunk_id 중복 제거 후 {len(chunk_map)}개")

    # 텍스트 기준 추가 중복 제거
    # 호 오파싱으로 동일 텍스트가 여러 chunk_id로 생성된 경우 제거
    seen_texts: set[str] = set()
    deduped: dict[str, dict] = {}
    for cid, chunk in chunk_map.items():
        # 공백 정규화 후 비교 — 띄어쓰기 차이로 동일 텍스트가 별개로 인식되는 문제 방지
        t = re.sub(r"\s+", " ", chunk.get("text", "")).strip()
        if t not in seen_texts:
            seen_texts.add(t)
            deduped[cid] = chunk
    print(f"   텍스트 중복 제거 후 {len(deduped)}개")
    chunk_map = deduped

    # ── 벡터 로드 ─────────────────────────
    print(f"\n📂 vectors.npz 로드: {VECTORS_PATH}")
    npz       = np.load(VECTORS_PATH)
    vectors   = npz["vectors"].astype(np.float32)
    chunk_ids = npz["chunk_ids"].tolist()
    print(f"   벡터 shape: {vectors.shape}")

    id_to_dense: dict[str, list[float]] = {}
    for cid, vec in zip(chunk_ids, vectors):
        id_to_dense[cid] = vec.tolist()

    # ── sparse 로드 ───────────────────────
    use_sparse = SPARSE_PATH.exists()
    id_to_sparse: dict[str, dict] = {}
    tokenizer = None

    if use_sparse:
        print(f"\n📂 sparse_weights.json 로드: {SPARSE_PATH}")
        with open(SPARSE_PATH, encoding="utf-8") as f:
            sparse_list: list[dict] = json.load(f)
        for cid, sw in zip(chunk_ids, sparse_list):
            id_to_sparse[cid] = sw
        print(f"   sparse 벡터: {len(id_to_sparse)}개")

        # FlagEmbedding 없이 토크나이저만 로드
        print(f"\n📦 토크나이저 로드: {EMBED_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        print(f"   토크나이저 vocab 크기: {tokenizer.vocab_size}")
    else:
        print(f"\n⚠️  sparse_weights.json 없음 → dense 단독 upsert")

    # ── upsert 대상 확정 ──────────────────
    common_ids = [cid for cid in chunk_map if cid in id_to_dense]
    print(f"\n   upsert 대상: {len(common_ids)}개")

    # ── Qdrant 컬렉션 준비 ────────────────
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        print(f"\n⚠️  컬렉션 '{COLLECTION}' 이미 존재 → 재생성합니다.")
        client.delete_collection(COLLECTION)

    if use_sparse:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense": VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )
        print(f"✅ 컬렉션 '{COLLECTION}' 생성 완료 (Dense + Sparse)")
    else:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"✅ 컬렉션 '{COLLECTION}' 생성 완료 (Dense only)")

    # ── 배치 upsert ───────────────────────
    print(f"\n⬆️  upsert 시작 (batch_size={BATCH_SIZE})...")
    points: list[PointStruct] = []

    for i, cid in enumerate(common_ids):
        chunk   = chunk_map[cid]
        payload = {
            "chunk_id"   : cid,
            "law_name"   : chunk.get("law_name",   ""),
            "article_id" : chunk.get("article_id", ""),
            "article"    : chunk.get("article",    ""),
            "category"   : chunk.get("category",   ""),
            "is_risk_ref": bool(chunk.get("is_risk_ref", False)),
            "chunk_text" : chunk.get("text",        ""),
            # source_full: evaluation gold_sources 매칭용
            "source_full": chunk.get("article",    ""),
        }

        if use_sparse and cid in id_to_sparse:
            sparse_vec = to_sparse_vector(id_to_sparse[cid], tokenizer)
            point = PointStruct(
                id=i,
                vector={
                    "dense" : id_to_dense[cid],
                    "sparse": sparse_vec,
                },
                payload=payload,
            )
        else:
            point = PointStruct(
                id=i,
                vector=id_to_dense[cid],
                payload=payload,
            )

        points.append(point)

        if len(points) == BATCH_SIZE:
            client.upsert(collection_name=COLLECTION, points=points)
            print(f"   [{i + 1}/{len(common_ids)}] upsert...", end="\r")
            points = []

    if points:
        client.upsert(collection_name=COLLECTION, points=points)

    # ── 확인 ─────────────────────────────
    count = client.count(collection_name=COLLECTION)
    print(f"\n✅ 완료: {count.count}개 포인트 저장됨")

    risk_count = client.count(
        collection_name=COLLECTION,
        count_filter={"must": [{"key": "is_risk_ref", "match": {"value": True}}]},
    )
    print(f"   is_risk_ref=True: {risk_count.count}개")


if __name__ == "__main__":
    main()