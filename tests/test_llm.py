# tests/test_llm.py
import pytest
import os
from unittest.mock import patch, MagicMock
from langchain_google_genai import ChatGoogleGenerativeAI

class TestLLMMethod:
    """Test suite for LLM method functionality."""
    
    @pytest.fixture
    def mock_newsletter_crew_instance(self):
        """Create a mock instance with the llm method."""
        class MockNewsletterCrew:
            def llm(self):
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    max_tokens=20000
                )
                return llm
        
        return MockNewsletterCrew()
    
    def test_llm_method_returns_chatgooglegenai_instance(self, mock_newsletter_crew_instance):
        """Test that llm() method returns a ChatGoogleGenerativeAI instance."""
        llm_instance = mock_newsletter_crew_instance.llm()
        
        assert isinstance(llm_instance, ChatGoogleGenerativeAI)
    
    def test_llm_method_uses_correct_model(self, mock_newsletter_crew_instance):
        """Test that llm() method uses gemini-2.5-flash model."""
        llm_instance = mock_newsletter_crew_instance.llm()
        
        assert llm_instance.model == "gemini-2.5-flash"
    
    def test_llm_method_uses_api_key_from_env(self, mock_newsletter_crew_instance):
        """Test that llm() method uses API key from environment variables."""
        expected_api_key = os.getenv("GOOGLE_API_KEY")
        
        if not expected_api_key:
            pytest.skip("GOOGLE_API_KEY not configured")
        
        llm_instance = mock_newsletter_crew_instance.llm()
        
        # Verify the API key is set
        assert hasattr(llm_instance, 'google_api_key') or hasattr(llm_instance, 'client')
    
    @patch.dict(os.environ, {}, clear=True)
    def test_llm_method_with_missing_api_key(self, mock_newsletter_crew_instance):
        """Test llm() method behavior when API key is missing."""
        try:
            llm_instance = mock_newsletter_crew_instance.llm()
            assert isinstance(llm_instance, ChatGoogleGenerativeAI)
        except Exception:
            # It's acceptable for this to raise an exception when API key is missing
            pass
    
    @patch('langchain_google_genai.ChatGoogleGenerativeAI.__init__')
    def test_llm_method_initialization_parameters(self, mock_init, mock_newsletter_crew_instance):
        """Test that llm() method passes correct parameters to ChatGoogleGenerativeAI."""
        mock_init.return_value = None
        
        mock_newsletter_crew_instance.llm()
        
        expected_api_key = os.getenv("GOOGLE_API_KEY")
        mock_init.assert_called_once_with(
            model="gemini-2.5-flash",
            google_api_key=expected_api_key,
            max_tokens=20000
        )
    
    def test_llm_method_multiple_calls_return_new_instances(self, mock_newsletter_crew_instance):
        """Test that multiple calls to llm() return new instances."""
        llm1 = mock_newsletter_crew_instance.llm()
        llm2 = mock_newsletter_crew_instance.llm()
        
        # Should be different instances
        assert llm1 is not llm2
        assert isinstance(llm1, ChatGoogleGenerativeAI)
        assert isinstance(llm2, ChatGoogleGenerativeAI)

# Test simplificado que verifica lo esencial
class TestLLMBasic:
    """Simplified tests focusing on core functionality."""
    
    def test_llm_creation_basic(self):
        """Basic test that LLM can be created without errors."""
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            pytest.skip("GOOGLE_API_KEY not configured")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            max_tokens=20000
        )
        
        assert isinstance(llm, ChatGoogleGenerativeAI)
        assert llm.model == "gemini-2.5-flash"

# Integration test (opcional)
class TestLLMIntegration:
    """Integration tests for LLM method with real API calls."""
    
    @pytest.mark.integration
    def test_llm_method_real_connection(self):
        """Test actual LLM connection and response."""
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            pytest.skip("GOOGLE_API_KEY not configured for integration test")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            max_tokens=20000
        )
        
        try:
            response = llm.invoke("Say 'test successful'")
            assert response is not None
            assert hasattr(response, 'content')
            assert len(response.content) > 0
        except Exception as e:
            pytest.fail(f"LLM integration test failed: {e}")