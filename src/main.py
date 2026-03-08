from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
import time

from arxiv_fetcher import ArxivFetcher
from bib_loader import load_library
from embedding_cache import LibraryEmbeddingCache
from emailer import build_email_html, build_email_subject, send_email
from embedder import SentenceTransformerEmbedder
from recommender import Recommender
from settings import load_settings, load_smtp_settings


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend new arXiv papers from local bib files.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--dry-run", action="store_true", help="Build the report but do not send email.")
    parser.add_argument(
        "--output-html",
        default=None,
        help="Optional override for the rendered HTML report path.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> int:
    configure_logging()
    args = parse_args()
    settings = load_settings(Path(args.config))

    if not settings.arxiv.categories:
        raise ValueError("config.yaml must define at least one arXiv category under arxiv.categories")

    library_papers, library_stats = load_library(settings.runtime.data_dir)
    if not library_papers:
        raise ValueError("No bib entries with abstracts were found under the configured data directory")

    fetcher = ArxivFetcher(
        categories=settings.arxiv.categories,
        max_candidates=settings.arxiv.max_candidates,
    )
    candidate_papers, fetch_stats = fetcher.fetch_new_papers()

    embedder = SentenceTransformerEmbedder(
        model_name=settings.embedding.model,
        batch_size=settings.embedding.batch_size,
    )
    library_cache = LibraryEmbeddingCache(
        cache_dir=settings.runtime.cache_dir,
        model_name=settings.embedding.model,
    )
    library_embedding_started_at = time.perf_counter()
    library_embeddings = library_cache.load_or_compute(library_papers, embedder)
    LOGGER.info("Library embedding stage finished in %.2f seconds", time.perf_counter() - library_embedding_started_at)
    recommender = Recommender(
        embedder=embedder,
        top_k_neighbors=settings.ranking.top_k_neighbors,
        max_results=settings.ranking.max_results,
    )
    recommendations, recommendation_stats = recommender.recommend(
        library_papers,
        candidate_papers,
        library_embeddings=library_embeddings,
    )
    LOGGER.info(
        "Pipeline stats | rss_new=%s rss_unique=%s fallback_used=%s fallback_count=%s fetched=%s after_dedup=%s threshold_filtered=%s final=%s",
        fetch_stats.rss_new_count,
        fetch_stats.rss_unique_count,
        fetch_stats.fallback_used,
        fetch_stats.fallback_candidate_count,
        fetch_stats.fetched_candidate_count,
        recommendation_stats.after_dedup_filter_count,
        recommendation_stats.threshold_filtered_count,
        recommendation_stats.final_recommendation_count,
    )

    generated_at = datetime.now(timezone.utc)
    html_body = build_email_html(
        recommendations=recommendations,
        library_stats=library_stats,
        fetch_stats=fetch_stats,
        recommendation_stats=recommendation_stats,
        include_pdf_links=settings.email.include_pdf_links,
        generated_at=generated_at,
    )

    output_html = Path(args.output_html) if args.output_html else settings.runtime.output_html
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_body, encoding="utf-8")
    LOGGER.info("Wrote HTML report to %s", output_html)

    if not recommendations and not settings.email.send_empty_email:
        LOGGER.info("No recommendations found; skipping email because send_empty_email is disabled")
        return 0

    if args.dry_run:
        LOGGER.info("Dry-run mode enabled; skipping SMTP send")
        return 0

    smtp_settings = load_smtp_settings()
    subject = build_email_subject(
        subject_prefix=settings.email.subject_prefix,
        recommendation_count=len(recommendations),
        generated_at=generated_at,
    )
    send_email(subject, html_body, smtp_settings)
    LOGGER.info("Sent recommendation email to %s", smtp_settings.recipient)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - top-level failure logging
        logging.getLogger(__name__).exception("Pipeline failed: %s", exc)
        raise
