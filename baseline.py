"""
Baseline: ChunkingStrategyComparator trên 3 tài liệu Shopee
Chạy: python3 baseline.py
"""
from src.chunking import ChunkingStrategyComparator
from pathlib import Path

FILES = [
    "data/shopee_dong_kiem.md",
    "data/shopee_phuong_thuc_tra_hang.md",
    "data/shopee_chinh_sach_tra_hang_hoan_tien.md",
]

comp = ChunkingStrategyComparator()

for f in FILES:
    text = Path(f).read_text()
    result = comp.compare(text, chunk_size=500)
    name = Path(f).stem
    print(f"\n=== {name} ===")
    for strategy, stats in result.items():
        print(f"  {strategy:12} | chunks={stats['count']:3} | avg_len={stats['avg_length']:6.1f}")
