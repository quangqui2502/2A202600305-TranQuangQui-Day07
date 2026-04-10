# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Trần Quang Quí
**Nhóm:** C401-F2
**Ngày:** 2026-04-10

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai text chunk có cosine similarity cao nghĩa là các vector embedding của chúng gần như cùng hướng trong không gian nhiều chiều — tức là hai đoạn văn có nội dung/ngữ nghĩa tương tự nhau, bất kể độ dài.

**Ví dụ HIGH similarity:**
- Sentence A: "Tôi muốn trả hàng và hoàn tiền."
- Sentence B: "Tôi muốn gửi yêu cầu hoàn trả sản phẩm."
- Tại sao tương đồng: Cả hai đều nói về cùng hành động (trả hàng/hoàn tiền), dùng từ ngữ đồng nghĩa — embedding nắm bắt được ý nghĩa chung.

**Ví dụ LOW similarity:**
- Sentence A: "Con mèo đang ngủ trên ghế sofa."
- Sentence B: "Tiền hoàn về ví ShopeePay mất bao lâu?"
- Tại sao khác: Hai câu hoàn toàn khác domain và chủ đề — vector hướng về hai vùng khác nhau trong không gian embedding.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity đo góc giữa hai vector nên không bị ảnh hưởng bởi độ dài văn bản — một đoạn dài và một đoạn ngắn cùng nói về một chủ đề vẫn có similarity cao. Euclidean distance bị ảnh hưởng bởi magnitude của vector, khiến văn bản dài luôn "xa hơn" văn bản ngắn dù nội dung tương đồng.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
> = ceil((10000 - 50) / (500 - 50))
> = ceil(9950 / 450)
> = ceil(22.11)
> **= 23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Với overlap=100: ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = **25 chunks** — tăng thêm 2 chunks. Overlap nhiều hơn giúp context quan trọng không bị cắt đứt giữa hai chunk liền kề, đặc biệt hữu ích với văn bản có câu dài hoặc thông tin trải dài qua nhiều dòng.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Shopee — Chính sách Trả hàng / Hoàn tiền (FAQ)

**Tại sao nhóm chọn domain này?**
> Shopee là sàn TMĐT phổ biến tại Việt Nam, chính sách trả hàng/hoàn tiền là nội dung người dùng thường xuyên cần tra cứu. Domain này có cấu trúc FAQ rõ ràng, nhiều điều kiện cụ thể dễ đánh giá độ chính xác của retrieval. Ngoài ra tài liệu tiếng Việt giúp nhóm thực hành RAG với ngôn ngữ thực tế.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | shopee_chinh_sach_tra_hang_hoan_tien.md | help.shopee.vn/portal/4/article/77251 | 27,287 | category: "policy", topic: "return_refund", lang: "vi" |
| 2 | shopee_dong_kiem.md | help.shopee.vn/portal/4/article/124982 | 9,948 | category: "faq", topic: "dong_kiem", lang: "vi" |
| 3 | shopee_huy_don_hoan_voucher.md | help.shopee.vn/portal/4/article/79296 | 7,641 | category: "faq", topic: "voucher_refund", lang: "vi" |
| 4 | shopee_phuong_thuc_tra_hang.md | help.shopee.vn/portal/4/article/189477 | 9,467 | category: "guide", topic: "return_method", lang: "vi" |
| 5 | shopee_quy_dinh_chung_tra_hang.md | help.shopee.vn | 9,014 | category: "policy", topic: "return_rules", lang: "vi" |
| 6 | shopee_thoi_gian_hoan_tien.md | help.shopee.vn/portal/4/article/189473 | 7,346 | category: "faq", topic: "refund_timeline", lang: "vi" |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| category | string | "policy", "faq", "guide" | Phân loại tài liệu — filter khi muốn chỉ tìm FAQ hoặc policy |
| topic | string | "return_refund", "dong_kiem", "refund_timeline" | Narrow scope retrieval theo chủ đề cụ thể |
| lang | string | "vi" | Hữu ích nếu sau này mở rộng sang tài liệu tiếng Anh |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| shopee_chinh_sach... | FixedSizeChunker | 47 | 497 | Không — cắt giữa điều khoản |
| shopee_chinh_sach... | SentenceChunker | 56 | 374 | Trung bình — câu tốt nhưng chunk nhỏ |
| shopee_chinh_sach... | RecursiveChunker | 66 | 317 | Trung bình — chunk quá nhỏ |
| shopee_dong_kiem | FixedSizeChunker | 18 | 497 | Không — cắt ngang giữa câu hỏi |
| shopee_dong_kiem | SentenceChunker | 10 | 809 | Tốt hơn — giữ được ý |
| shopee_dong_kiem | RecursiveChunker | 25 | 322 | Kém — quá nhỏ |
| shopee_phuong_thuc | FixedSizeChunker | 17 | 492 | Không — cắt giữa bước hướng dẫn |
| shopee_phuong_thuc | SentenceChunker | 11 | 686 | Tốt — giữ đoạn hướng dẫn |
| shopee_phuong_thuc | RecursiveChunker | 26 | 290 | Kém — quá vụn |

### Strategy Của Tôi

**Loại:** Custom — `DocumentStructureChunker`

**Mô tả cách hoạt động:**
> Split đầu tiên theo markdown headers (`##`, `###`) và bold numbered questions (`**N.**`) bằng regex lookahead. Nếu một section vẫn lớn hơn `max_chunk_size` (1200 ký tự), tiếp tục split theo numbered sub-items (`1.`, `1.1.`, ...). Các đoạn nhỏ liền kề được gom vào buffer cho đến khi đầy. Kết quả: mỗi chunk tương ứng một Q&A hoặc một điều khoản hoàn chỉnh.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu Shopee FAQ có cấu trúc rõ ràng: mỗi câu hỏi được đánh số (`**1.**`, `**2.**`) và mỗi section có header `##`. Splitting theo ranh giới cấu trúc này đảm bảo mỗi chunk chứa đúng 1 câu hỏi + câu trả lời đầy đủ — tránh hiện tượng câu trả lời bị cắt ngang giữa chừng như FixedSizeChunker.

**Code snippet:**
```python
class DocumentStructureChunker:
    def __init__(self, max_chunk_size: int = 1200) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> list[str]:
        import re
        header_pattern = re.compile(
            r'(?=^#{1,3} |\n#{1,3} |(?:^|\n)\*\*\d+\.)',
            re.MULTILINE,
        )
        sections = header_pattern.split(text)
        chunks = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) <= self.max_chunk_size:
                chunks.append(section)
                continue
            sub_pattern = re.compile(r'(?=(?:^|\n)\d+(?:\.\d+)*\. )', re.MULTILINE)
            sub_sections = sub_pattern.split(section)
            buffer = ""
            for sub in sub_sections:
                sub = sub.strip()
                if not sub:
                    continue
                if len(buffer) + len(sub) + 1 <= self.max_chunk_size:
                    buffer = (buffer + "\n" + sub).strip()
                else:
                    if buffer:
                        chunks.append(buffer)
                    buffer = sub
            if buffer:
                chunks.append(buffer)
        return [c for c in chunks if c]
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| shopee_dong_kiem | SentenceChunker (best baseline) | 10 | 809 | Tốt — nhưng gộp nhiều câu hỏi |
| shopee_dong_kiem | **DocumentStructureChunker** | **13** | **621** | **Tốt hơn — mỗi chunk = 1 Q&A** |
| shopee_chinh_sach... | SentenceChunker (best baseline) | 56 | 374 | Trung bình — chunk nhỏ, thiếu context |
| shopee_chinh_sach... | **DocumentStructureChunker** | **18** | **1166** | **Tốt hơn — giữ nguyên điều khoản** |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Trần Quang Quí | DocumentStructureChunker + OpenAI | 9/10 (5/5 relevant, avg score 0.628) | Chunk bám sát cấu trúc Q&A, context coherent, không bị cắt giữa điều khoản | Multi-aspect query (Q3) score thấp 0.59 vì định nghĩa và hướng dẫn nằm ở 2 chunk khác nhau |
| Nhữ Gia Bách | SemanticChunker | 8/10 | Chunk đúng chủ đề, score distribution rõ | Thiếu thông tin số liệu cụ thể khi chunk tách rời context |
| Đoàn Nam Sơn | Parent-Child Chunking | 9/10 (5/5 relevant, avg score 0.66) | Chunk hoạt động rất tốt, cắt đúng theo pattern Q&A, không bị cắt giữa điều khoản, rất phù hợp với tài liệu có cấu trúc rõ ràng | Với tài liệu không có cấu trúc rõ ràng thì có thể không hiệu quả |
| Hoàng Vĩnh Giang | Custom Recursive Strategy | 8/10 | Chunking dựa theo cấu trúc tài liệu, đảm bảo tính toàn vẹn của thông tin đoạn văn | Với những đoạn dài, chunk có thể vượt quá lượng ký tự cho phép của mô hình embedding |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Parent-Child Chunking (Sơn) và DocumentStructureChunker (Quí) cho kết quả tốt nhất với avg score lần lượt 0.66 và 0.628 — cả hai đều khai thác cấu trúc Q&A có sẵn của tài liệu Shopee thay vì split cơ học. Điểm chung: chunk coherent, mỗi chunk = 1 đơn vị ý nghĩa hoàn chỉnh. SemanticChunker và Custom Recursive cũng tốt nhưng dễ bị ảnh hưởng bởi độ dài chunk không đồng đều. Kết luận: với domain FAQ có cấu trúc rõ ràng, **structure-aware chunking luôn tốt hơn fixed-size hay sentence-based**.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng `re.split(r'(?<=[.!?])\s+', text.strip())` để tách câu — lookbehind đảm bảo chỉ split sau dấu câu kết thúc thật sự, không split giữa số thập phân hay viết tắt. Các câu được gom vào `current_chunk` list cho đến khi đạt `max_sentences_per_chunk`, rồi join bằng space và append vào kết quả. Phần dư cuối cùng (chưa đủ số câu) được flush nốt vào chunk cuối.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Base case: nếu `len(text) <= chunk_size` thì trả về `[text.strip()]` ngay. Nếu không, lấy separator đầu tiên trong danh sách ưu tiên `["\n\n", "\n", ". ", " ", ""]` để split text thành các `parts`, rồi gom vào `buffer` cho đến khi buffer + piece vượt chunk_size — lúc đó flush buffer và đệ quy `_split(buffer, remaining_separators[1:])` để xử lý tiếp với separator nhỏ hơn. Nếu hết separator, hard-split theo ký tự.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Store dùng Python list `_store` lưu các record dict gồm `id`, `doc_id`, `content`, `embedding`, `metadata`. Khi `add_documents`, mỗi `Document` được embed bằng `_embedding_fn(doc.content)` rồi append vào list. Khi `search`, embed query, tính dot product với từng record (embedding đã normalize nên dot product ≈ cosine similarity), sort descending, trả về top-k.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` filter trước: duyệt `_store`, giữ lại record có metadata khớp toàn bộ key-value trong `metadata_filter`, sau đó mới gọi `_search_records` trên tập đã lọc. `delete_document` dùng list comprehension lọc bỏ tất cả record có `doc_id` trùng, trả về `True` nếu size giảm.

### KnowledgeBaseAgent

**`answer`** — approach:
> Gọi `store.search(question, top_k)` lấy top-k chunks, join content bằng newline thành `context`. Prompt được cấu trúc: system instruction ("Use the context below... if not in context, say you don't know") + context block + question + "Answer:" để guide LLM bám sát tài liệu. Truyền prompt vào `llm_fn` và trả về kết quả trực tiếp.

### Test Results

```
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.0.2, pluggy-1.6.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================== 42 passed in 0.05s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Tôi muốn trả hàng và hoàn tiền. | Tôi muốn gửi yêu cầu hoàn trả sản phẩm. | high | 0.7141 | Yes |
| 2 | Shopee hoàn tiền trong 24 giờ. | Tiền sẽ về tài khoản sau một ngày. | high | 0.5101 | Yes |
| 3 | Đồng kiểm là kiểm tra ngoại quan khi nhận hàng. | Phí vận chuyển được hoàn lại sau 3-5 ngày. | low | 0.4067 | Yes |
| 4 | Con mèo đang ngủ trên ghế sofa. | Tiền hoàn về ví ShopeePay mất bao lâu? | low | 0.2048 | Yes |
| 5 | Mã giảm giá có được hoàn không? | Voucher có được trả lại khi hủy đơn không? | high | 0.5428 | Yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 2 bất ngờ nhất — "24 giờ" và "sau một ngày" là cùng nghĩa nhưng score chỉ 0.51, không cao như kỳ vọng. Điều này cho thấy embeddings nắm bắt ngữ nghĩa tốt ở cấp độ từ vựng và cụm từ, nhưng suy luận đồng nghĩa số học ("24 giờ" = "một ngày") vẫn còn hạn chế. Pair 5 cũng thú vị: "mã giảm giá" và "voucher" là cùng khái niệm nhưng dùng hai từ khác nhau — score 0.54 cho thấy model đã học được sự tương đồng này từ dữ liệu huấn luyện.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Tôi có bao nhiêu ngày để gửi yêu cầu trả hàng hoàn tiền? | 15 ngày kể từ lúc đơn hàng được cập nhật trạng thái Giao hàng thành công. Riêng thực phẩm tươi sống/đông lạnh: 24 giờ. |
| 2 | Tiền hoàn về ví ShopeePay mất bao lâu? | 24 giờ (với điều kiện Ví ShopeePay vẫn hoạt động bình thường và còn liên kết với tài khoản Shopee). |
| 3 | Đồng kiểm là gì và tôi được làm gì khi đồng kiểm? | Đồng kiểm là kiểm tra ngoại quan và số lượng sản phẩm khi nhận hàng. Được: kiểm tra bên ngoài, số lượng. Không được: mở tem chống hàng giả, dùng thử sản phẩm, làm hư hại sản phẩm. |
| 4 | Nếu trả hàng theo hình thức tự sắp xếp, tôi có được hoàn phí vận chuyển không? | Có. Bạn thanh toán trước phí trả hàng, Shopee hoàn lại trong 3–5 ngày làm việc: đơn Shopee Mall hoàn tiền mặt; đơn ngoài Mall hoàn Shopee Xu (25,000 xu nếu cùng tỉnh, 40,000 xu nếu khác tỉnh). |
| 5 | Mã giảm giá có được hoàn lại khi tôi trả hàng toàn bộ đơn không? | Tùy theo quy định hoàn mã giảm giá của Shopee. Voucher có thể được/không được hoàn tùy điều kiện cụ thể (loại voucher, lý do trả hàng). |

### Kết Quả Của Tôi

**Chunker:** `DocumentStructureChunker(max_chunk_size=1200)` | **Embedder:** `text-embedding-3-small` | **LLM:** `gpt-4o-mini`
**Documents loaded:** 6 files → 59 chunks

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Tôi có bao nhiêu ngày để gửi yêu cầu trả hàng hoàn tiền? | Mục 1.2 quy định chung — thời gian gửi yêu cầu trả hàng/hoàn tiền | 0.644 | Yes | "Bạn có 15 ngày kể từ lúc đơn hàng cập nhật giao thành công. Thực phẩm tươi sống/đông lạnh: 24 giờ." |
| 2 | Tiền hoàn về ví ShopeePay mất bao lâu? | Lưu ý hoàn tiền — điều kiện ví ShopeePay hoạt động bình thường | 0.624 | Yes | "24 giờ, với điều kiện ví cần ở trạng thái bình thường và còn liên kết với tài khoản Shopee." |
| 3 | Đồng kiểm là gì và tôi được làm gì khi đồng kiểm? | Câu 6 — hướng dẫn cách đồng kiểm với bưu tá (Shipper) | 0.590 | Yes | "Kiểm tra ngoại quan và số lượng sản phẩm. Không được mở tem, dùng thử, làm hư hại sản phẩm." |
| 4 | Nếu trả hàng theo hình thức tự sắp xếp, tôi có được hoàn phí vận chuyển không? | Mục 1.4 — lưu ý khi chọn hình thức tự sắp xếp | 0.655 | Yes | "Cần thanh toán trước phí trả hàng. Shopee hoàn lại trong 3–5 ngày làm việc sau khi yêu cầu được chấp nhận." |
| 5 | Mã giảm giá có được hoàn lại khi tôi trả hàng toàn bộ đơn không? | Mục 2 quy định chung — bảng hoàn mã giảm giá/Shopee Xu | 0.629 | Yes | "Mã giảm giá sẽ được hoàn lại khi trả hàng toàn bộ đơn." |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

### So Sánh Kết Quả Trong Nhóm (Exercise 3.4)

**Strategy nào cho retrieval tốt nhất? Tại sao?**
> Parent-Child (Sơn, avg 0.66) và DocumentStructureChunker (Quí, avg 0.628) dẫn đầu — cả hai đều khai thác ranh giới cấu trúc Q&A tự nhiên của tài liệu. FixedSizeChunker và SentenceChunker cho điểm thấp hơn vì cắt ngang giữa câu hỏi và câu trả lời, làm giảm coherence của chunk.

**Có query nào strategy A tốt hơn B nhưng ngược lại ở query khác không?**
> Q2 (ShopeePay mất bao lâu): DocumentStructureChunker (score 0.624) tốt hơn SentenceChunker (score ~0.55) vì thông tin nằm trong bảng markdown — SentenceChunker không nhận ra ranh giới bảng. Ngược lại, Q3 (Đồng kiểm): SentenceChunker (score ~0.61) gom được câu 1 và câu 6 vào cùng chunk hơn DocumentStructureChunker (0.59) vì chunk theo câu không bị chia theo header.

**Metadata filtering có giúp ích không?**
> Có — đặc biệt với Q2: filter `topic="refund_timeline"` loại bỏ các chunk từ 5 file khác chứa từ "hoàn" nhưng sai chủ đề, giúp top-1 luôn là chunk từ `shopee_thoi_gian_hoan_tien.md`. Không filter thì top-1 có thể bị nhiễu bởi chunk "phí trả hàng" (cùng có từ "hoàn", score cao nhưng sai nội dung).

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Từ Sơn (Parent-Child Chunking): ý tưởng lưu 2 cấp chunk — chunk lớn (parent) để giữ context đầy đủ và chunk nhỏ (child) để retrieve chính xác hơn — giải quyết được trade-off giữa precision và context mà single-level chunking không làm được. Từ Giang (Custom Recursive): khi chunk vượt giới hạn embedding model thì cần fallback strategy rõ ràng, không nên để lỗi im lặng.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Một nhóm dùng metadata filtering theo danh mục sản phẩm để narrow scope retrieval trước khi search — cách tiếp cận này tăng precision đáng kể so với search toàn bộ store, đặc biệt khi knowledge base có nhiều domain overlap. Bài học: thiết kế metadata schema kỹ từ đầu giúp ích rất nhiều ở giai đoạn retrieval sau này.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Sẽ thêm bước tiền xử lý tài liệu: loại bỏ các dòng URL dài, link markdown, và ký hiệu bảng rỗng trước khi chunk — những nội dung này tạo ra noise chunk làm loãng embedding space. Ngoài ra sẽ thử `DocumentStructureChunker` với `max_chunk_size` nhỏ hơn (600-800) để mỗi chunk chứa đúng 1 Q&A thay vì có thể gom 2-3 câu hỏi liền kề, từ đó tăng precision cho retrieval.

### Failure Analysis

**Query retrieval kém nhất:** Q3 — "Đồng kiểm là gì và tôi được làm gì khi đồng kiểm?" (score top-1: 0.590, thấp nhất trong 5 queries)

**Nguyên nhân:** Query hỏi 2 thứ cùng lúc (định nghĩa + hành động), nhưng tài liệu tách thành câu 1 (định nghĩa) và câu 6 (cách làm) ở hai chunk khác nhau. Top-1 chunk retrieve được câu 6 (cách làm) với score 0.59 — đủ để trả lời phần "làm gì", nhưng bỏ sót phần định nghĩa đầy đủ ở câu 1 (score 0.47, chỉ top-2).

**Phân loại lỗi:** Multi-aspect query — một câu hỏi cần thông tin từ nhiều chunk khác nhau.

**Đề xuất cải thiện:**
1. Tăng `top_k=5` thay vì 3 để lấy cả chunk định nghĩa lẫn chunk hướng dẫn
2. Query rewriting: tách "Đồng kiểm là gì?" và "Tôi được làm gì khi đồng kiểm?" thành 2 sub-queries, merge context trước khi trả lời
3. Chunk câu 1 và câu 6 vào cùng 1 chunk bằng cách tăng `max_chunk_size` đủ để gộp định nghĩa + hướng dẫn

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 9 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **85 / 100** |
