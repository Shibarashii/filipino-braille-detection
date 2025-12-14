"""
LLM-based Text Correction for Braille OCR
Uses free AI APIs to fix missing/wrong letters using context
"""
import requests
import json
from typing import Optional, Dict, List
import time


class LLMTextCorrector:
    """Uses free LLM APIs to correct Braille OCR text with context"""

    def __init__(self, api_choice='groq', api_key=None):
        """
        Initialize LLM corrector

        Args:
            api_choice: 'groq', 'huggingface', 'ollama', or 'together'
            api_key: API key (free tier available for most)
        """
        self.api_choice = api_choice.lower()
        self.api_key = api_key
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # seconds between requests

        # API endpoints
        self.endpoints = {
            'groq': 'https://api.groq.com/openai/v1/chat/completions',
            'huggingface': 'https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf',
            'together': 'https://api.together.xyz/v1/chat/completions',
        }

        print(f"✓ LLM Text Corrector initialized (using {self.api_choice})")
        if not api_key and api_choice != 'ollama':
            print(
                f"⚠️  No API key provided. Set it with: corrector.set_api_key('your_key')")

    def set_api_key(self, api_key: str):
        """Set API key"""
        self.api_key = api_key
        print(f"✓ API key set for {self.api_choice}")

    def _rate_limit(self):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def correct_text(self, text: str, language='en', context_hint=None) -> Dict:
        """
        Correct text using LLM

        Args:
            text: Raw OCR text with potential errors
            language: 'en' for English, 'tl' for Filipino/Tagalog, 'both' for mixed
            context_hint: Optional hint about content (e.g., "educational text", "story")

        Returns:
            Dict with corrected_text, original_text, changes, confidence
        """
        if not text or not text.strip():
            return {
                'corrected_text': '',
                'original_text': text,
                'changes': [],
                'confidence': 0.0,
                'method': 'none'
            }

        # Choose correction method based on API
        if self.api_choice == 'ollama':
            return self._correct_with_ollama(text, language, context_hint)
        elif self.api_choice == 'groq':
            return self._correct_with_groq(text, language, context_hint)
        elif self.api_choice == 'huggingface':
            return self._correct_with_huggingface(text, language, context_hint)
        elif self.api_choice == 'together':
            return self._correct_with_together(text, language, context_hint)
        else:
            print(f"⚠️  Unknown API choice: {self.api_choice}")
            return {
                'corrected_text': text,
                'original_text': text,
                'changes': [],
                'confidence': 0.0,
                'method': 'none'
            }

    def _build_prompt(self, text: str, language: str, context_hint: Optional[str]) -> str:
        """Build correction prompt"""

        lang_context = ""
        if language == 'tl' or language == 'filipino':
            lang_context = "This is Filipino/Tagalog text. "
        elif language == 'both' or language == 'mixed':
            lang_context = "This is mixed Filipino and English text. "
        else:
            lang_context = "This is English text. "

        context_info = f"Context: {context_hint}. " if context_hint else ""

        prompt = f"""You are helping correct text from Braille OCR (optical character recognition). The text may have missing or incorrect letters due to detection errors.

{lang_context}{context_info}

Your task:
1. Read the text carefully and understand the intended meaning
2. Fix missing letters (e.g., "brille" → "braille", "nable" → "enable")
3. Fix wrong letters (e.g., "brille" → "braille")
4. Maintain the original structure (line breaks, spacing)
5. For Filipino text, ensure proper spelling (e.g., "kumian" → "kumain")
6. Keep numbers and punctuation as-is
7. Do NOT add new words or change the meaning
8. Do NOT explain your corrections, just return the corrected text

Original text with errors:
\"\"\"
{text}
\"\"\"

Return ONLY the corrected text, nothing else. No explanations, no markdown, just the corrected text."""

        return prompt

    def _correct_with_groq(self, text: str, language: str, context_hint: Optional[str]) -> Dict:
        """Correct using Groq API (FREE, very fast)"""

        if not self.api_key:
            return self._fallback_result(text, "No API key")

        self._rate_limit()

        prompt = self._build_prompt(text, language, context_hint)

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': 'llama-3.3-70b-versatile',  # Fast and free
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.1,  # Low temperature for consistent corrections
            'max_tokens': 1000
        }

        try:
            response = requests.post(
                self.endpoints['groq'],
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                corrected = result['choices'][0]['message']['content'].strip()

                # Clean up any markdown or explanations
                corrected = self._clean_response(corrected)

                changes = self._find_changes(text, corrected)

                return {
                    'corrected_text': corrected,
                    'original_text': text,
                    'changes': changes,
                    'confidence': 0.9 if changes else 1.0,
                    'method': 'groq'
                }
            else:
                print(f"⚠️  Groq API error: {response.status_code}")
                return self._fallback_result(text, f"API error {response.status_code}")

        except Exception as e:
            print(f"⚠️  Groq API exception: {e}")
            return self._fallback_result(text, str(e))

    def _correct_with_together(self, text: str, language: str, context_hint: Optional[str]) -> Dict:
        """Correct using Together API (FREE tier available)"""

        if not self.api_key:
            return self._fallback_result(text, "No API key")

        self._rate_limit()

        prompt = self._build_prompt(text, language, context_hint)

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': 'meta-llama/Llama-3-8b-chat-hf',
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.1,
            'max_tokens': 1000
        }

        try:
            response = requests.post(
                self.endpoints['together'],
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                corrected = result['choices'][0]['message']['content'].strip()
                corrected = self._clean_response(corrected)
                changes = self._find_changes(text, corrected)

                return {
                    'corrected_text': corrected,
                    'original_text': text,
                    'changes': changes,
                    'confidence': 0.85 if changes else 1.0,
                    'method': 'together'
                }
            else:
                return self._fallback_result(text, f"API error {response.status_code}")

        except Exception as e:
            return self._fallback_result(text, str(e))

    def _correct_with_huggingface(self, text: str, language: str, context_hint: Optional[str]) -> Dict:
        """Correct using HuggingFace Inference API (FREE)"""

        if not self.api_key:
            return self._fallback_result(text, "No API key")

        self._rate_limit()

        prompt = self._build_prompt(text, language, context_hint)

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'inputs': prompt,
            'parameters': {
                'max_new_tokens': 1000,
                'temperature': 0.1,
                'return_full_text': False
            }
        }

        try:
            response = requests.post(
                self.endpoints['huggingface'],
                headers=headers,
                json=data,
                timeout=60  # HF can be slower
            )

            if response.status_code == 200:
                result = response.json()
                corrected = result[0]['generated_text'].strip()
                corrected = self._clean_response(corrected)
                changes = self._find_changes(text, corrected)

                return {
                    'corrected_text': corrected,
                    'original_text': text,
                    'changes': changes,
                    'confidence': 0.8 if changes else 1.0,
                    'method': 'huggingface'
                }
            else:
                return self._fallback_result(text, f"API error {response.status_code}")

        except Exception as e:
            return self._fallback_result(text, str(e))

    def _correct_with_ollama(self, text: str, language: str, context_hint: Optional[str]) -> Dict:
        """Correct using local Ollama (100% FREE, runs locally)"""

        prompt = self._build_prompt(text, language, context_hint)

        data = {
            'model': 'llama3.2',  # Or any model you have installed
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': 0.1
            }
        }

        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                corrected = result['response'].strip()
                corrected = self._clean_response(corrected)
                changes = self._find_changes(text, corrected)

                return {
                    'corrected_text': corrected,
                    'original_text': text,
                    'changes': changes,
                    'confidence': 0.85 if changes else 1.0,
                    'method': 'ollama'
                }
            else:
                return self._fallback_result(text, f"Ollama error {response.status_code}")

        except requests.exceptions.ConnectionError:
            print("⚠️  Ollama not running. Start with: ollama serve")
            return self._fallback_result(text, "Ollama not running")
        except Exception as e:
            return self._fallback_result(text, str(e))

    def _clean_response(self, text: str) -> str:
        """Clean LLM response (remove markdown, explanations, etc.)"""

        # Remove markdown code blocks
        text = text.replace('```', '').strip()

        # If response starts with explanation, try to extract just the corrected text
        lines = text.split('\n')

        # Look for common patterns
        if 'corrected text:' in text.lower():
            # Extract text after "corrected text:"
            parts = text.lower().split('corrected text:')
            if len(parts) > 1:
                text = parts[1].strip()

        # Remove quotes if the entire text is quoted
        text = text.strip('"\'')

        return text.strip()

    def _find_changes(self, original: str, corrected: str) -> List[Dict]:
        """Find what changed between original and corrected"""
        changes = []

        orig_words = original.split()
        corr_words = corrected.split()

        for i, (orig, corr) in enumerate(zip(orig_words, corr_words)):
            if orig != corr:
                changes.append({
                    'position': i,
                    'original': orig,
                    'corrected': corr
                })

        return changes

    def _fallback_result(self, text: str, error: str) -> Dict:
        """Return original text when API fails"""
        return {
            'corrected_text': text,
            'original_text': text,
            'changes': [],
            'confidence': 0.0,
            'method': 'fallback',
            'error': error
        }


def get_free_api_instructions():
    """Print instructions for getting free API keys"""

    instructions = """
    ╔══════════════════════════════════════════════════════════════╗
    ║          FREE AI API OPTIONS FOR BRAILLE CORRECTION          ║
    ╚══════════════════════════════════════════════════════════════╝
    
    🚀 RECOMMENDED: Groq (Fastest, Free Tier)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Website: https://console.groq.com
    • Model: Llama 3.3 70B (very fast and accurate)
    • Free tier: 30 requests/minute, 14,400/day
    • Setup:
      1. Sign up at console.groq.com
      2. Go to API Keys
      3. Create new API key
      4. Use: corrector = LLMTextCorrector('groq', 'your_api_key')
    
    🏠 BEST FOR LOCAL: Ollama (100% Free, Private)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Website: https://ollama.com
    • No API key needed, runs on your computer
    • Setup:
      1. Install: curl -fsSL https://ollama.com/install.sh | sh
      2. Pull model: ollama pull llama3.2
      3. Start: ollama serve
      4. Use: corrector = LLMTextCorrector('ollama')
    • Pros: Private, unlimited, no cost
    • Cons: Requires decent hardware (8GB+ RAM)
    
    🤗 Together AI (Free Tier Available)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Website: https://api.together.xyz
    • Model: Llama 3 8B
    • Free tier: $25 credit on signup
    • Setup:
      1. Sign up at api.together.xyz
      2. Get API key from dashboard
      3. Use: corrector = LLMTextCorrector('together', 'your_api_key')
    
    🤗 HuggingFace (Free Tier)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Website: https://huggingface.co
    • Model: Llama 2 7B
    • Free tier: Rate limited but free
    • Setup:
      1. Sign up at huggingface.co
      2. Go to Settings → Access Tokens
      3. Create token with "read" access
      4. Use: corrector = LLMTextCorrector('huggingface', 'your_token')
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    💡 Quick Start (Groq):
    
    1. Get API key: console.groq.com
    2. In your code:
       
       from llm_text_corrector import LLMTextCorrector
       
       corrector = LLMTextCorrector('groq', 'your_api_key_here')
       result = corrector.correct_text(
           "brille nable blind peple to rad",
           language='en'
       )
       print(result['corrected_text'])
       # Output: "braille enable blind people to read"
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    print(instructions)


# Example usage
if __name__ == "__main__":
    # Show instructions
    get_free_api_instructions()

    # Example with Groq (replace with your key)
    print("\n" + "="*60)
    print("EXAMPLE CORRECTIONS")
    print("="*60)

    # Simulated example (won't actually work without API key)
    corrector = LLMTextCorrector('groq')

    test_texts = [
        "brille nable blind peple to rad",
        "kumian ako ng tinapay",  # Filipino
        "the qick brown fox jmps"
    ]

    for text in test_texts:
        print(f"\nOriginal: {text}")
        print(f"Expected: [Corrected version with LLM]")
        print(f"Note: Set API key with: corrector.set_api_key('your_key')")
