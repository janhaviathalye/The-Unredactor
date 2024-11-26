import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from unredactor import extract_features
import pandas as pd

@pytest.fixture
def sample_data():
    """Fixture for sample data."""
    return {
        "cleaned_context": [
            "This is a story about REDACTED who travels to Europe.",
            "The main character REDACTED is a compelling lead."
        ]
    }

def test_extract_features(sample_data):
    """Test the extract_features function."""
    features, vectorizer = extract_features(pd.DataFrame(sample_data), fit=True)
    assert features.shape[0] == 2  # Two contexts
    assert vectorizer is not None
    assert isinstance(vectorizer, TfidfVectorizer)
