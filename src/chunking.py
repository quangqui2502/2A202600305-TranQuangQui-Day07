from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        sentences = re.split(r'(?<=[\.\!\?])\s+|(?<=\.)\n', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # No separators left — force split by character
        if not remaining_separators:
            results = []
            for i in range(0, len(current_text), self.chunk_size):
                results.append(current_text[i : i + self.chunk_size])
            return results

        sep = remaining_separators[0]
        rest = remaining_separators[1:]

        if sep == "":
            # Character-level fallback
            return self._split(current_text, rest)

        parts = current_text.split(sep)
        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Part itself may be too long — recurse with remaining separators
                if len(part) > self.chunk_size:
                    chunks.extend(self._split(part, rest))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks if chunks else [current_text]


class DocumentStructureChunker:
    """
    Markdown/HTML-aware chunking — splits on structural boundaries:
    heading (#, ##, ###), table blocks, and bullet list blocks.

    Strategy:
        1. Split on heading lines (## Title) → each section = 1 chunk candidate.
        2. If a section is still too large, fall back to paragraph (\n\n) splitting.
        3. Preserve the heading as prefix of each child chunk so context is not lost.
    """

    def __init__(self, max_chunk_size: int = 600) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Split on heading lines (lines starting with #)
        heading_pattern = re.compile(r'(?=^#{1,6}\s)', re.MULTILINE)
        sections = heading_pattern.split(text)
        sections = [s.strip() for s in sections if s.strip()]

        # If no headings found, fall back to paragraph splitting
        if not sections:
            sections = [text]

        chunks: list[str] = []
        for section in sections:
            if len(section) <= self.max_chunk_size:
                chunks.append(section)
            else:
                # Section too large — split on double newline, keep heading prefix
                lines = section.split("\n")
                heading = lines[0] if lines[0].startswith("#") else ""
                sub_sections = section.split("\n\n")
                current = ""
                for sub in sub_sections:
                    candidate = (current + "\n\n" + sub).strip() if current else sub.strip()
                    if len(candidate) <= self.max_chunk_size:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current)
                        # Prepend heading for context
                        current = (heading + "\n\n" + sub).strip() if heading and not sub.startswith("#") else sub.strip()
                if current:
                    chunks.append(current)

        return [c for c in chunks if c]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size).chunk(text),
            "by_sentences": SentenceChunker().chunk(text),
            "recursive": RecursiveChunker(chunk_size=chunk_size).chunk(text),
        }
        result = {}
        for name, chunks in strategies.items():
            avg_len = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
            result[name] = {
                "count": len(chunks),
                "avg_length": round(avg_len, 1),
                "chunks": chunks,
            }
        return result
