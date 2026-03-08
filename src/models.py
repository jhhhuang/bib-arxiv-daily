from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class LibraryPaper:
    title: str
    abstract: str
    source_file: str
    bib_key: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None

    @property
    def embedding_text(self) -> str:
        return f"{self.title}\n\n{self.abstract}"


@dataclass(frozen=True)
class CandidatePaper:
    title: str
    abstract: str
    authors: tuple[str, ...]
    entry_id: str
    pdf_url: str | None
    published: datetime | None
    categories: tuple[str, ...] = field(default_factory=tuple)
    doi: str | None = None
    arxiv_id: str | None = None

    @property
    def embedding_text(self) -> str:
        return f"{self.title}\n\n{self.abstract}"

    @property
    def arxiv_url(self) -> str:
        if self.entry_id.startswith("http://") or self.entry_id.startswith("https://"):
            return self.entry_id
        return f"https://arxiv.org/abs/{self.entry_id}"


@dataclass(frozen=True)
class NeighborMatch:
    title: str
    similarity: float


@dataclass(frozen=True)
class Recommendation:
    candidate: CandidatePaper
    score: float
    neighbors: tuple[NeighborMatch, ...]


@dataclass(frozen=True)
class LibraryLoadStats:
    files_scanned: int
    entries_total: int
    entries_with_abstract: int
    duplicates_removed: int
    skipped_missing_title: int
    skipped_missing_abstract: int


@dataclass(frozen=True)
class ArxivFetchStats:
    rss_new_count: int
    rss_unique_count: int
    fetched_candidate_count: int
    fallback_used: bool = False
    fallback_window_hours: int = 0
    fallback_candidate_count: int = 0


@dataclass(frozen=True)
class RecommendationStats:
    input_candidate_count: int
    after_dedup_filter_count: int
    threshold_filtered_count: int
    final_recommendation_count: int
