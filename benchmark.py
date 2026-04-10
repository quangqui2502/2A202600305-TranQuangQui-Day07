"""
Group benchmark: DocumentStructureChunker + OpenAI embeddings
Nhóm: C401-F2 | Domain: Shopee Trả hàng / Hoàn tiền

Chạy:
    EMBEDDING_PROVIDER=openai python3 benchmark.py
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.chunking import DocumentStructureChunker
from src.embeddings import OpenAIEmbedder, _mock_embed
from src.models import Document
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent

load_dotenv(override=False)

# ── Data files ────────────────────────────────────────────────────────────────
SHOPEE_FILES = [
    ("data/shopee_chinh_sach_tra_hang_hoan_tien.md", "policy", "return_refund"),
    ("data/shopee_dong_kiem.md",                     "faq",    "dong_kiem"),
    ("data/shopee_huy_don_hoan_voucher.md",          "faq",    "voucher_refund"),
    ("data/shopee_phuong_thuc_tra_hang.md",          "guide",  "return_method"),
    ("data/shopee_quy_dinh_chung_tra_hang.md",       "policy", "return_rules"),
    ("data/shopee_thoi_gian_hoan_tien.md",           "faq",    "refund_timeline"),
]

# ── Benchmark queries + gold answers ─────────────────────────────────────────
QUERIES = [
    {
        "id": "Q1",
        "query": "Tôi có bao nhiêu ngày để gửi yêu cầu trả hàng hoàn tiền?",
        "gold": "15 ngày kể từ lúc đơn hàng được cập nhật Giao hàng thành công "
                "(thực phẩm tươi sống/đông lạnh: 24 giờ).",
    },
    {
        "id": "Q2",
        "query": "Tiền hoàn về ví ShopeePay mất bao lâu?",
        "gold": "24 giờ (với điều kiện Ví ShopeePay vẫn hoạt động bình thường).",
    },
    {
        "id": "Q3",
        "query": "Đồng kiểm là gì và tôi được làm gì khi đồng kiểm?",
        "gold": "Kiểm tra ngoại quan và số lượng sản phẩm. "
                "Không được mở tem, dùng thử, làm hư hại sản phẩm.",
    },
    {
        "id": "Q4",
        "query": "Nếu trả hàng theo hình thức tự sắp xếp, tôi có được hoàn phí vận chuyển không?",
        "gold": "Có. Shopee hoàn trong 3–5 ngày làm việc "
                "(tiền mặt cho Shopee Mall, Shopee Xu cho đơn ngoài Mall).",
    },
    {
        "id": "Q5",
        "query": "Mã giảm giá có được hoàn lại khi tôi trả hàng toàn bộ đơn không?",
        "gold": "Tùy điều kiện: voucher có thể được/không được hoàn theo quy định Shopee.",
    },
]


def load_and_chunk_documents(chunker: DocumentStructureChunker) -> list[Document]:
    docs: list[Document] = []
    for path_str, category, topic in SHOPEE_FILES:
        path = Path(path_str)
        if not path.exists():
            print(f"  [SKIP] {path_str} not found")
            continue
        raw = path.read_text(encoding="utf-8")
        chunks = chunker.chunk(raw)
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                id=f"{path.stem}_chunk{i}",
                content=chunk,
                metadata={
                    "source": path_str,
                    "category": category,
                    "topic": topic,
                    "chunk_index": i,
                    "lang": "vi",
                },
            ))
        print(f"  {path.stem}: {len(chunks)} chunks")
    return docs


def run_benchmark(store: EmbeddingStore, agent: KnowledgeBaseAgent) -> None:
    relevant_count = 0
    for q in QUERIES:
        print(f"\n{'─'*60}")
        print(f"[{q['id']}] {q['query']}")
        print(f"  Gold: {q['gold']}")

        results = store.search(q["query"], top_k=3)
        print("  Top-3 retrieved chunks:")
        for rank, r in enumerate(results, 1):
            src = r["metadata"].get("source", "?")
            topic = r["metadata"].get("topic", "?")
            preview = r["content"][:120].replace("\n", " ")
            print(f"    {rank}. [{topic}] score={r['score']:.4f} | {preview}...")

        answer = agent.answer(q["query"], top_k=3)
        print(f"  Agent answer: {answer[:300].replace(chr(10), ' ')}")

        # Simple relevance check: top-1 score > 0.3 treated as relevant
        if results and results[0]["score"] > 0.3:
            relevant_count += 1

    print(f"\n{'='*60}")
    print(f"Relevant in top-3: {relevant_count} / {len(QUERIES)}")


def real_llm(prompt: str) -> str:
    """Call OpenAI chat to generate answer from retrieved context."""
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM error: {e}]"


def main() -> None:
    provider = os.getenv("EMBEDDING_PROVIDER", "mock").strip().lower()

    if provider == "openai":
        try:
            embedder = OpenAIEmbedder()
            llm_fn = real_llm
            print(f"Embedding backend: {embedder._backend_name}")
            print("LLM backend: gpt-4o-mini")
        except Exception as e:
            print(f"OpenAI init failed ({e}), falling back to mock")
            embedder = _mock_embed
            llm_fn = lambda p: "[mock LLM]"
    else:
        embedder = _mock_embed
        llm_fn = lambda p: "[mock LLM — set EMBEDDING_PROVIDER=openai for real answers]"
        print("Embedding backend: mock (set EMBEDDING_PROVIDER=openai to use OpenAI)")

    chunker = DocumentStructureChunker(max_chunk_size=1200)

    print("\n── Loading & chunking Shopee documents ──")
    docs = load_and_chunk_documents(chunker)
    print(f"Total chunks: {len(docs)}")

    store = EmbeddingStore(collection_name="shopee_faq", embedding_fn=embedder)
    print("\nEmbedding & indexing...")
    store.add_documents(docs)
    print(f"Store size: {store.get_collection_size()} chunks")

    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)

    print("\n── Running 5 benchmark queries ──")
    run_benchmark(store, agent)


if __name__ == "__main__":
    main()
