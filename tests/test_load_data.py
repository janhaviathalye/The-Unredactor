import pytest
import pandas as pd
from unredactor import load_data

@pytest.fixture
def sample_data(tmp_path):
    """Create a sample dataset for testing."""
    data = pd.DataFrame({
        "split": ["training", "validation"],
        "name": ["John Doe", "Jane Smith"],
        "context": [
            "This is a story about ██████ who travels to Europe.",
            "The main character ███████████ is a compelling lead."
        ]
    })
    file_path = tmp_path / "test_data.tsv"
    data.to_csv(file_path, sep="\t", index=False, header=False)
    return file_path

def test_load_data(sample_data):
    """Test the load_data function."""
    data = load_data(sample_data)
    assert "cleaned_context" in data.columns
    assert len(data) == 2
    assert "REDACTED" in data['cleaned_context'].iloc[0]
