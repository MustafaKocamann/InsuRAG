import json
from pathlib import Path
from typing import List, Dict, Any, Optional

DEFAULT_TESTS_PATH = Path(__file__).parent / "tests.jsonl"


class TestQuestion:
    """Represents a single evaluation test question."""

    def __init__(
        self,
        question: str,
        keywords: Optional[List[str]] = None,
        reference_answer: str = "",
        category: str = "general",
        **kwargs: Any,
    ):
        self.question = question
        self.keywords = keywords or []
        self.reference_answer = reference_answer
        self.category = category
        self.extra = kwargs

    def __repr__(self) -> str:
        return f"TestQuestion(question={self.question!r}, category={self.category!r})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "keywords": self.keywords,
            "reference_answer": self.reference_answer,
            "category": self.category,
            **self.extra,
        }


# Alias for backward compatibility
Test = TestQuestion


def load_tests(file_path: Optional[str | Path] = None) -> List[TestQuestion]:
    """
    Load evaluation test cases from a JSONL file into a list of TestQuestion objects.
    """
    path = Path(file_path) if file_path else DEFAULT_TESTS_PATH
    tests: List[TestQuestion] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                tests.append(TestQuestion(**data))
    return tests
