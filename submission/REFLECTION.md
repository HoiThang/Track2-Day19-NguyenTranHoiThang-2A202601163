# Reflection — Lab 19

**Tên:** Nguyễn Trần Hội Thắng
**Cohort:** A20-K2
**Path đã chạy:** lite + docker

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

**Kết quả:**
- `exact` queries: BM25 thắng (96.7%) vì có technical terms verbatim trong corpus
- `paraphrase` queries: Vector thắng nhưng vẫn yếu (24-33%) với bge-small-en; hybrid giúp cải thiện nhẹ
- `mixed` queries: **Hybrid thắng tuyệt đối (100%)** — kết hợp cả exact term và paraphrase

**Khi nào KHÔNG dùng hybrid:**
1. Query quá ngắn (< 3 tokens) — không đủ signal cho BM25
2. Khi cần latency cực thấp (keyword-only < 5ms vs hybrid ~60ms)
3. Corpus có ít vocabulary overlap — BM25 signal yếu, vector chưa đủ tốt

---

## Điều ngạc nhiên nhất khi làm lab này

**Filtered ANN recall cliff**: post-filter về 0.00 recall khi filter selectivity ~4% — đây là silent failure rất nguy hiểm trong production vì không có error, chỉ có kết quả tệ dần.

**On-demand feature vs stored feature**: `amount_vs_avg = amount / avg_amount_7d` là feature fraud mạnh nhất nhưng không thể materialize — đây là use case mà Feast on-demand feature view giải quyết rất elegant.

---

## Bonus challenge

- [ ] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _<tên đồng đội nếu có>_
