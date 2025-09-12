import pytest
from dotenv import load_dotenv

@pytest.fixture(scope="session", autouse=True)
def setup_env():
    """Load environment variables for all tests."""
    load_dotenv()