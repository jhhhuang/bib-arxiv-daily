from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from emailer import build_email_html
from models import ArxivFetchStats, CandidatePaper, LibraryLoadStats, NeighborMatch, Recommendation, RecommendationStats


class EmailerTest(unittest.TestCase):
    def test_build_email_html_includes_pdf_link_and_neighbor_titles(self) -> None:
        recommendation = Recommendation(
            candidate=CandidatePaper(
                title="Graph Signal Learning",
                abstract="Learning on graph-structured data with spectral methods.",
                authors=("Author One", "Author Two"),
                entry_id="http://arxiv.org/abs/2501.00001v1",
                pdf_url="http://arxiv.org/pdf/2501.00001v1",
                published=datetime(2025, 1, 1),
                arxiv_id="2501.00001v1",
            ),
            score=0.9123,
            neighbors=(
                NeighborMatch(title="Graph Neural Networks", similarity=0.95),
                NeighborMatch(title="Spectral Graph Theory", similarity=0.88),
            ),
        )
        stats = LibraryLoadStats(
            files_scanned=2,
            entries_total=10,
            entries_with_abstract=8,
            duplicates_removed=1,
            skipped_missing_title=0,
            skipped_missing_abstract=1,
        )
        fetch_stats = ArxivFetchStats(
            rss_new_count=4,
            rss_unique_count=4,
            fetched_candidate_count=4,
        )
        recommendation_stats = RecommendationStats(
            input_candidate_count=4,
            after_dedup_filter_count=3,
            threshold_filtered_count=0,
            final_recommendation_count=1,
        )

        html = build_email_html(
            [recommendation],
            library_stats=stats,
            fetch_stats=fetch_stats,
            recommendation_stats=recommendation_stats,
            include_pdf_links=True,
            generated_at=datetime(2025, 1, 1),
        )

        self.assertIn("Graph Signal Learning", html)
        self.assertIn("Graph Neural Networks", html)
        self.assertIn("http://arxiv.org/pdf/2501.00001v1", html)
        self.assertIn("Library papers with abstracts: 8", html)
        self.assertIn("RSS new papers:", html)
        self.assertIn("After dedupe / already-in-library filter:", html)
        self.assertIn("Threshold filtered:", html)

    def test_build_email_html_shows_lookback_window_for_manual_weekly_run(self) -> None:
        stats = LibraryLoadStats(
            files_scanned=2,
            entries_total=10,
            entries_with_abstract=8,
            duplicates_removed=1,
            skipped_missing_title=0,
            skipped_missing_abstract=1,
        )
        fetch_stats = ArxivFetchStats(
            rss_new_count=0,
            rss_unique_count=0,
            fetched_candidate_count=12,
            query_mode="lookback",
            lookback_days=7,
        )
        recommendation_stats = RecommendationStats(
            input_candidate_count=12,
            after_dedup_filter_count=10,
            threshold_filtered_count=0,
            final_recommendation_count=0,
        )

        html = build_email_html(
            [],
            library_stats=stats,
            fetch_stats=fetch_stats,
            recommendation_stats=recommendation_stats,
            include_pdf_links=True,
            generated_at=datetime(2025, 1, 8),
        )

        self.assertIn("Query window:", html)
        self.assertIn("last 7 days via export API", html)
        self.assertIn("Fetched candidates:", html)
        self.assertNotIn("RSS new papers:", html)


if __name__ == "__main__":
    unittest.main()
