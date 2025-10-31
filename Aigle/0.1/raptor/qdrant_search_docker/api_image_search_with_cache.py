import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import uvicorn
import time
from cache_manager import CacheManager


# ========= 資料模型 =========
class SearchRequest(BaseModel):
    query_text: str = Field(..., description="搜索關鍵字", min_length=1)
    embedding_type: str = Field(..., description="搜索類型: summary 或 text")
    type: Optional[str] = Field(None, description="集合類型: audio/video/document/image")
    filename: Optional[List[str]] = Field(None, description="文件名列表")
    source: Optional[str] = Field(None, description="圖像格式: jpg/png/jpeg/gif/bmp 等")
    limit: int = Field(5, description="返回結果數量", ge=1, le=100)


class SearchResult(BaseModel):
    score: float
    id: str
    payload: dict


class SearchResponse(BaseModel):
    success: bool
    total: int
    results: List[SearchResult]


class IndexResponse(BaseModel):
    success: bool
    message: str
    indexes: Optional[dict] = None


# ========= 初始化 =========
app = FastAPI(title="圖像相似度搜索 API", version="1.0.0")

client = None
model = None
collection_name = "images"

cm = CacheManager(
    host="192.168.157.165",
    port=7000,
    password="dht888888",
    max_connections=1000,
    ttl=3600,
    ttl_multiplier=1e-2,
    is_cluster=True
)


# ========= 啟動事件 =========
@app.on_event("startup")
async def startup_event():
    global client, model
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

    print(f"🔌 連接 Qdrant ({qdrant_host}:{qdrant_port}) ...")
    client = AsyncQdrantClient(host=qdrant_host, port=qdrant_port)

    try:
        collection_info = await client.get_collection(collection_name)
        print(f"✅ Collection '{collection_name}' 已存在")
    except Exception:
        print(f"⚠️ Collection 不存在，正在創建...")
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        print(f"✅ 成功創建 collection: {collection_name}")

    print("🤖 載入模型 BAAI/bge-m3 ...")
    model = SentenceTransformer("BAAI/bge-m3")
    print("✅ 模型載入完成")


# ========= Cache 搜尋 =========
@cm.cache(semantic=True)
async def cached_search(collection_name, query_vector, query_filter, limit):
    """✅ 支援 async 的快取搜尋"""
    results = await client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False
    )
    return results


# ========= 篩選條件 =========
def build_filter(embedding_type, type_value=None, filenames=None, source=None) -> Optional[Filter]:
    must_conditions = [
        FieldCondition(key="status", match=MatchValue(value="active")),
        FieldCondition(key="embedding_type", match=MatchValue(value=embedding_type))
    ]
    if type_value:
        must_conditions.append(FieldCondition(key="type", match=MatchValue(value=type_value)))
    if filenames:
        if len(filenames) == 1:
            must_conditions.append(FieldCondition(key="filename", match=MatchValue(value=filenames[0])))
        else:
            must_conditions.append(FieldCondition(key="filename", match=MatchAny(any=filenames)))
    if source:
        must_conditions.append(FieldCondition(key="source", match=MatchValue(value=source)))
    return Filter(must=must_conditions)


# ========= 搜尋 API =========
@app.post("/image_search", response_model=SearchResponse)
async def search_images(request: SearchRequest):
    try:
        start = time.perf_counter()

        if request.embedding_type not in ["summary", "text"]:
            raise HTTPException(status_code=400, detail="embedding_type 必須是 'summary' 或 'text'")

        query_filter = build_filter(
            embedding_type=request.embedding_type,
            type_value=request.type,
            filenames=request.filename,
            source=request.source
        )

        query_vector = model.encode(request.query_text).tolist()

        results = await cached_search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=request.limit
        )

        formatted_results = [
            SearchResult(score=res.score, id=str(res.id), payload=res.payload)
            for res in results
        ]

        end = time.perf_counter()
        print(f"[TIMED] /image_search took {end - start:.3f}s")

        return SearchResponse(success=True, total=len(formatted_results), results=formatted_results)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失敗: {str(e)}")


# ========= 健康檢查 =========
@app.get("/health")
async def health_check():
    try:
        info = await client.get_collection(collection_name)  
        return {"status": "healthy", "vectors_count": info.vectors_count}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api_image_search_with_cache:app", host="0.0.0.0", port=8814, reload=True)

