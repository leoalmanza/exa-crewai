# tests/test_api_key.py
import pytest
import os

def test_google_api_key_exists():
    """Test that GOOGLE_API_KEY environment variable is configured."""

    api_key = os.getenv("GOOGLE_API_KEY")
    
    assert api_key is not None, "GOOGLE_API_KEY not found in environment"
    assert len(api_key) > 20, "API key appears too short to be valid"
    print(f"✅ API Key found: {api_key[:20]}...")

def test_api_key_format():
    """Test basic API key format validation."""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if api_key:
        assert isinstance(api_key, str)
        assert not api_key.isspace()
        assert len(api_key.strip()) == len(api_key)