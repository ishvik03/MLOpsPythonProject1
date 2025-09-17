import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference import load_pipeline




# Any function starting with test_ will be automatically discovered and run by pytest.
def test_inference_pipeline():
    pipe = load_pipeline()
    result = pipe("I love MLOps!")

    # Check result structure
    assert isinstance(result, list)
    assert "label" in result[0]
    assert "score" in result[0]

    # Optional: check prediction makes sense
    assert result[0]["score"] > 0.5
