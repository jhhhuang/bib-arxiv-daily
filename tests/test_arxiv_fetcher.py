from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arxiv_fetcher import ArxivFetcher


class FakeFeedParser:
    def parse(self, url: str):
        self.last_url = url
        return SimpleNamespace(
            feed=SimpleNamespace(title="arXiv query results"),
            entries=[
                {"id": "oai:arXiv.org:2501.00001", "arxiv_announce_type": "new"},
                {"id": "oai:arXiv.org:2501.00002", "arxiv_announce_type": "replace"},
                {"id": "oai:arXiv.org:2501.00003", "arxiv_announce_type": "new"},
            ],
        )


class FakeArxivResult:
    def __init__(self, entry_id: str, title: str):
        self.entry_id = f"http://arxiv.org/abs/{entry_id}v1"
        self.title = title
        self.summary = f"Abstract for {title}"
        self.authors = [SimpleNamespace(name="Author One"), SimpleNamespace(name="Author Two")]
        self.pdf_url = f"http://arxiv.org/pdf/{entry_id}v1"
        self.categories = ["cs.LG"]
        self.published = datetime(2025, 1, 1)
        self.doi = None


class FakeArxivClient:
    def __init__(self, module, *args, **kwargs):
        self.module = module
        self.seen_batches = []
        self.seen_queries = []

    def results(self, search):
        if search.id_list is not None:
            self.seen_batches.append(tuple(search.id_list))
            return [FakeArxivResult(item, f"Paper {item}") for item in search.id_list]
        self.seen_queries.append(search)
        return list(self.module.query_results)


class FakeArxivModule:
    class Search:
        def __init__(self, id_list=None, query=None, max_results=None, sort_by=None, sort_order=None):
            self.id_list = None if id_list is None else list(id_list)
            self.query = query
            self.max_results = max_results
            self.sort_by = sort_by
            self.sort_order = sort_order

    class SortCriterion:
        SubmittedDate = "submittedDate"

    class SortOrder:
        Descending = "descending"

    def __init__(self, query_results=None):
        self.created_client = None
        self.query_results = list(query_results or [])

    def Client(self, *args, **kwargs):
        self.created_client = FakeArxivClient(self, *args, **kwargs)
        return self.created_client


class FakeEmptyFeedParser:
    def parse(self, url: str):
        self.last_url = url
        return SimpleNamespace(
            feed=SimpleNamespace(title="arXiv query results"),
            entries=[],
        )


class ArxivFetcherTest(unittest.TestCase):
    def test_fetch_new_papers_uses_only_new_rss_entries(self) -> None:
        fake_feedparser = FakeFeedParser()
        fake_arxiv = FakeArxivModule()
        fetcher = ArxivFetcher(
            categories=("cs.LG", "cs.AI"),
            max_candidates=10,
            feedparser_module=fake_feedparser,
            arxiv_module=fake_arxiv,
        )

        papers, stats = fetcher.fetch_new_papers()

        self.assertEqual("https://rss.arxiv.org/atom/cs.LG+cs.AI", fake_feedparser.last_url)
        self.assertEqual(2, len(papers))
        self.assertEqual(2, stats.rss_new_count)
        self.assertEqual(2, stats.rss_unique_count)
        self.assertEqual(2, stats.fetched_candidate_count)
        self.assertEqual(("2501.00001", "2501.00003"), fake_arxiv.created_client.seen_batches[0])
        self.assertEqual("2501.00001v1", papers[0].arxiv_id)
        self.assertEqual("2501.00003v1", papers[1].arxiv_id)

    def test_fetch_new_papers_falls_back_to_export_api_when_rss_is_empty(self) -> None:
        fake_feedparser = FakeEmptyFeedParser()
        fallback_results = [
            FakeArxivResult("2501.10001", "Fallback Paper One"),
            FakeArxivResult("2501.10002", "Fallback Paper Two"),
        ]
        fallback_results[0].published = datetime(2025, 1, 7, 12, 0, tzinfo=timezone.utc)
        fallback_results[1].published = datetime(2025, 1, 8, 6, 0, tzinfo=timezone.utc)
        fake_arxiv = FakeArxivModule(query_results=fallback_results)
        fetcher = ArxivFetcher(
            categories=("cs.LG", "cs.AI"),
            max_candidates=10,
            feedparser_module=fake_feedparser,
            arxiv_module=fake_arxiv,
            now_fn=lambda: datetime(2025, 1, 8, 12, 0, tzinfo=timezone.utc),
        )

        papers, stats = fetcher.fetch_new_papers()

        self.assertEqual("https://rss.arxiv.org/atom/cs.LG+cs.AI", fake_feedparser.last_url)
        self.assertEqual(2, len(papers))
        self.assertTrue(stats.fallback_used)
        self.assertEqual(24, stats.fallback_window_hours)
        self.assertEqual(2, stats.fallback_candidate_count)
        self.assertEqual(2, stats.fetched_candidate_count)
        self.assertEqual("Fallback Paper Two", papers[0].title)
        search = fake_arxiv.created_client.seen_queries[0]
        self.assertEqual(10, search.max_results)
        self.assertEqual("submittedDate", search.sort_by)
        self.assertEqual("descending", search.sort_order)
        self.assertEqual(
            "submittedDate:[20250107120000 TO 20250108120000] AND (cat:cs.LG OR cat:cs.AI)",
            search.query,
        )


if __name__ == "__main__":
    unittest.main()
