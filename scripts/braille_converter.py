"""
Robust Braille to Text Converter with Error Correction
Supports Filipino (Tagalog) and English with bilingual spell checking
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import re
from difflib import SequenceMatcher

try:
    from spellchecker import SpellChecker
    SPELLCHECK_AVAILABLE = True
except ImportError:
    SPELLCHECK_AVAILABLE = False
    print("⚠️  Install pyspellchecker for better corrections: pip install pyspellchecker")

try:
    from scripts.llm import LLMTextCorrector
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️  LLM correction not available. Install llm_text_corrector.py for AI-powered corrections")


@dataclass
class BrailleDetection:
    """Stores information about a detected Braille character"""
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center_x: float
    center_y: float


@dataclass
class CorrectionInfo:
    """Information about text corrections made"""
    original: str
    corrected: str
    confidence: float
    # 'spellcheck_en', 'spellcheck_tl', 'context', 'gap_filled', etc.
    method: str


class FilipinoEnglishSpellChecker:
    """Bilingual spell checker for Filipino and English"""

    def __init__(self):
        self.en_checker = None
        self.tl_checker = None

        if SPELLCHECK_AVAILABLE:
            # English checker
            self.en_checker = SpellChecker(language='en')

            # Filipino/Tagalog checker - we'll build a custom one
            self.tl_checker = SpellChecker(language=None)  # Start empty
            self._load_filipino_dictionary()

    def _load_filipino_dictionary(self):
        """Load common Filipino/Tagalog words"""
        # Common Filipino words
        filipino_words = [
            # Pronouns
            'ako', 'ikaw', 'siya', 'kami', 'kayo', 'sila', 'tayo',
            'ko', 'mo', 'niya', 'namin', 'ninyo', 'nila', 'atin',

            # Common verbs
            'kumain', 'uminom', 'maglaro', 'matulog', 'gumising',
            'pumunta', 'umuwi', 'magbasa', 'magsulat', 'makinig',
            'tumingin', 'tumakbo', 'lumakad', 'umakyat', 'bumaba',
            'bumili', 'magtinda', 'magtrabaho', 'mag-aral', 'magturo',

            # Common nouns
            'bahay', 'paaralan', 'eskwela', 'silid', 'kuwarto',
            'kusina', 'banyo', 'sala', 'hardin', 'kalye',
            'bata', 'lalaki', 'babae', 'tao', 'pamilya',
            'ama', 'ina', 'anak', 'kapatid', 'lolo', 'lola',
            'araw', 'gabi', 'umaga', 'hapon', 'tanghali',
            'pagkain', 'tubig', 'kape', 'tinapay', 'kanin',
            'ulam', 'gulay', 'prutas', 'isda', 'karne',
            'libro', 'papel', 'lapis', 'bolpen', 'mesa',
            'upuan', 'pinto', 'bintana', 'ilaw', 'salamin',

            # Common adjectives
            'maganda', 'pangit', 'mabuti', 'masama', 'malaki',
            'maliit', 'mataba', 'payat', 'mataas', 'mababa',
            'mahaba', 'maikli', 'malayo', 'malapit', 'mainit',
            'malamig', 'basa', 'tuyo', 'bago', 'luma',

            # Common adverbs/particles
            'na', 'pa', 'ay', 'ng', 'sa', 'ni', 'kay',
            'para', 'dahil', 'kasi', 'pero', 'at', 'o',
            'kung', 'kapag', 'habang', 'nang', 'noon',
            'ngayon', 'mamaya', 'bukas', 'kahapon',

            # Questions
            'ano', 'sino', 'saan', 'kailan', 'bakit', 'paano',
            'ilan', 'alin', 'kanino',

            # Numbers (Filipino)
            'isa', 'dalawa', 'tatlo', 'apat', 'lima',
            'anim', 'pito', 'walo', 'siyam', 'sampu',

            # Common phrases
            'oo', 'hindi', 'opo', 'salamat', 'walang', 'anuman',
            'pasensya', 'paumanhin', 'mabuhay', 'ingat',

            # Education-related
            'guro', 'estudyante', 'klase', 'leksyon', 'takdang',
            'aralin', 'pagsusulit', 'eksamen', 'marka', 'grado',
            'proyekto', 'presentasyon', 'talakayan', 'pag-aaral',

            # Special characters
            'enye', 'ñ',
        ]

        # Load words with frequency
        word_freq = {word: 100 for word in filipino_words}
        self.tl_checker.word_frequency.load_words(filipino_words)

    def check(self, word: str) -> Tuple[bool, str, str]:
        """
        Check spelling in both languages

        Returns:
            (is_correct, correction, language)
        """
        word_lower = word.lower()

        # Check English first
        if self.en_checker and word_lower not in self.en_checker:
            en_correction = self.en_checker.correction(word_lower)
            en_similarity = SequenceMatcher(
                None, word_lower, en_correction).ratio()

            # Check Filipino
            tl_in_dict = word_lower in self.tl_checker or word_lower in self.tl_checker.word_frequency

            if not tl_in_dict:
                # Try Filipino correction
                tl_correction = self.tl_checker.correction(word_lower)
                tl_similarity = SequenceMatcher(
                    None, word_lower, tl_correction).ratio() if tl_correction else 0

                # Choose best correction
                if en_similarity > tl_similarity and en_similarity > 0.6:
                    return (False, en_correction, 'en')
                elif tl_similarity > 0.6:
                    return (False, tl_correction, 'tl')
                else:
                    # Word might be correct in either language
                    return (True, word_lower, 'unknown')
            else:
                # Found in Filipino dictionary
                return (True, word_lower, 'tl')
        else:
            # Found in English dictionary
            return (True, word_lower, 'en')

    def get_candidates(self, word: str) -> Dict[str, List[str]]:
        """Get correction candidates from both languages"""
        candidates = {}

        if self.en_checker:
            en_cands = self.en_checker.candidates(word.lower())
            if en_cands:
                candidates['en'] = sorted(
                    en_cands,
                    key=lambda c: SequenceMatcher(
                        None, word.lower(), c).ratio(),
                    reverse=True
                )[:3]

        if self.tl_checker:
            tl_cands = self.tl_checker.candidates(word.lower())
            if tl_cands:
                candidates['tl'] = sorted(
                    tl_cands,
                    key=lambda c: SequenceMatcher(
                        None, word.lower(), c).ratio(),
                    reverse=True
                )[:3]

        return candidates


class RobustBrailleConverter:
    """Converts YOLO Braille detections to text with bilingual error correction"""

    def __init__(
        self,
        line_height_threshold=50,
        word_gap_threshold=30,
        char_gap_threshold=25,
        min_confidence=0.15,
        enable_spellcheck=True,
        enable_gap_detection=True,
        bilingual=True
    ):
        """
        Initialize robust converter for Filipino Grade 1 Braille

        Args:
            line_height_threshold: Max vertical distance for same line (pixels)
            word_gap_threshold: Min horizontal distance for word spacing (pixels)
            char_gap_threshold: Expected spacing between characters (pixels)
            min_confidence: Minimum confidence to trust a detection
            enable_spellcheck: Use spell checking for corrections
            enable_gap_detection: Detect and flag potential missing characters
            bilingual: Enable Filipino and English spell checking
        """
        self.line_height_threshold = line_height_threshold
        self.word_gap_threshold = word_gap_threshold
        self.char_gap_threshold = char_gap_threshold
        self.min_confidence = min_confidence
        self.enable_spellcheck = enable_spellcheck
        self.enable_gap_detection = enable_gap_detection
        self.bilingual = bilingual

        # Initialize bilingual spell checker
        self.spell_checker = None
        if enable_spellcheck and SPELLCHECK_AVAILABLE:
            self.spell_checker = FilipinoEnglishSpellChecker()
            print("✓ Bilingual spell checker initialized (Filipino + English)")

        self.corrections = []  # Track all corrections made
        self.low_confidence_words = []  # Track uncertain words

    def convert_results_to_text(
        self,
        results,
        apply_corrections=True,
        return_metadata=False
    ) -> str:
        """
        Convert YOLO results to readable text with bilingual corrections

        Args:
            results: YOLO prediction results
            apply_corrections: Apply spell checking and corrections
            return_metadata: Return additional info about corrections

        Returns:
            Converted text string (or dict if return_metadata=True)
        """
        if not results or len(results) == 0:
            return "" if not return_metadata else {"text": "", "corrections": [], "confidence": 1.0}

        # Reset tracking
        self.corrections = []
        self.low_confidence_words = []

        # Extract detections
        detections = self._extract_detections(results[0])

        if not detections:
            return "" if not return_metadata else {"text": "", "corrections": [], "confidence": 1.0}

        # Sort into lines and reading order
        lines = self._organize_into_lines(detections)

        # Detect potential gaps in each line
        if self.enable_gap_detection:
            lines = self._detect_and_flag_gaps(lines)

        # Convert each line to text
        text_lines = []
        line_confidences = []

        for line in lines:
            line_result = self._convert_line_to_text(
                line, track_confidence=True)
            line_text = line_result['text']
            line_conf = line_result['confidence']

            if line_text.strip():
                # Apply corrections if enabled
                if apply_corrections and self.enable_spellcheck:
                    line_text = self._correct_line(line_text, line_conf)

                text_lines.append(line_text)
                line_confidences.append(line_conf)

        final_text = "\n".join(text_lines)
        avg_confidence = np.mean(line_confidences) if line_confidences else 0.0

        # Apply LLM correction if enabled (after spell check)
        if apply_corrections and self.enable_llm_correction and self.llm_corrector and final_text.strip():
            print("\n🤖 Applying LLM-based context correction...")
            llm_result = self.llm_corrector.correct_text(
                final_text,
                language='both' if self.bilingual else 'en'
            )

            if llm_result['changes']:
                print(f"✓ LLM made {len(llm_result['changes'])} corrections")
                for change in llm_result['changes']:
                    print(
                        f"  '{change['original']}' → '{change['corrected']}'")

                self.llm_corrections = llm_result['changes']
                final_text = llm_result['corrected_text']

        if return_metadata:
            return {
                "text": final_text,
                "raw_text": "\n".join([l['text'] for l in [self._convert_line_to_text(line, track_confidence=True) for line in lines]]),
                "corrections": self.corrections,
                "llm_corrections": self.llm_corrections,
                "low_confidence_words": self.low_confidence_words,
                "average_confidence": avg_confidence,
                "num_detections": len(detections),
                "num_lines": len(lines)
            }

        return final_text

    def _extract_detections(self, result) -> List[BrailleDetection]:
        """Extract detection information from YOLO result"""
        detections = []

        if len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_names = result.names

        for i in range(len(boxes)):
            bbox = boxes[i]
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            class_id = int(classes[i])
            class_name = class_names[class_id]
            confidence = confidences[i]

            detections.append(BrailleDetection(
                class_name=class_name,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                center_x=center_x,
                center_y=center_y
            ))

        return detections

    def _organize_into_lines(self, detections: List[BrailleDetection]) -> List[List[BrailleDetection]]:
        """Organize detections into lines based on vertical position"""
        if not detections:
            return []

        # Sort by vertical position first
        sorted_detections = sorted(detections, key=lambda d: d.center_y)

        lines = []
        current_line = [sorted_detections[0]]

        for detection in sorted_detections[1:]:
            # Check if on same line as previous
            if abs(detection.center_y - current_line[-1].center_y) <= self.line_height_threshold:
                current_line.append(detection)
            else:
                # New line
                lines.append(current_line)
                current_line = [detection]

        # Add last line
        if current_line:
            lines.append(current_line)

        # Sort each line left to right
        for line in lines:
            line.sort(key=lambda d: d.center_x)

        return lines

    def _detect_and_flag_gaps(self, lines: List[List[BrailleDetection]]) -> List[List[BrailleDetection]]:
        """Detect unusually large gaps that might indicate missing characters"""
        for line in lines:
            if len(line) < 2:
                continue

            # Calculate gaps between consecutive detections
            gaps = []
            for i in range(len(line) - 1):
                gap = line[i + 1].center_x - line[i].center_x
                gaps.append(gap)

            if not gaps:
                continue

            # Find median gap (typical character spacing)
            median_gap = np.median(gaps)

            # Flag unusually large gaps (potential missing characters)
            for i, gap in enumerate(gaps):
                # If gap is much larger than typical but smaller than word gap
                if gap > median_gap * 1.8 and gap < self.word_gap_threshold:
                    # Mark as potential missing character location
                    print(
                        f"⚠️  Posibleng nawawalang character sa pagitan ng '{line[i].class_name}' at '{line[i+1].class_name}' (gap: {gap:.0f}px)")

        return lines

    def _convert_line_to_text(self, line: List[BrailleDetection], track_confidence=False) -> dict:
        """Convert a line of Braille detections to text"""
        if not line:
            return {"text": "", "confidence": 1.0, "word_confidences": []}

        text = []
        word_confidences = []
        current_word_chars = []
        current_word_confs = []

        i = 0
        capitalize_next = False
        number_mode = False

        while i < len(line):
            detection = line[i]
            class_name = detection.class_name
            confidence = detection.confidence

            # Handle special indicators
            if class_name == "capital":
                capitalize_next = True
                i += 1
                continue

            if class_name == "number":
                number_mode = True
                i += 1
                continue

            # Check for word gaps
            if i > 0:
                prev_detection = line[i - 1]
                gap = detection.center_x - prev_detection.center_x

                if gap > self.word_gap_threshold:
                    # End current word
                    if current_word_chars:
                        word = "".join(current_word_chars)
                        avg_conf = np.mean(
                            current_word_confs) if current_word_confs else 1.0

                        # Track low confidence words
                        if avg_conf < self.min_confidence + 0.15:
                            self.low_confidence_words.append({
                                "word": word,
                                "confidence": avg_conf
                            })

                        text.append(word)
                        word_confidences.append(avg_conf)
                        current_word_chars = []
                        current_word_confs = []

                    text.append(" ")
                    number_mode = False  # Space ends number mode

            # Convert character
            char = self._convert_character(
                class_name, capitalize_next, number_mode)

            if char:
                current_word_chars.append(char)
                current_word_confs.append(confidence)

            # Reset flags
            capitalize_next = False

            i += 1

        # Add last word
        if current_word_chars:
            word = "".join(current_word_chars)
            avg_conf = np.mean(
                current_word_confs) if current_word_confs else 1.0

            if avg_conf < self.min_confidence + 0.15:
                self.low_confidence_words.append({
                    "word": word,
                    "confidence": avg_conf
                })

            text.append(word)
            word_confidences.append(avg_conf)

        final_text = "".join(text)
        avg_confidence = np.mean(word_confidences) if word_confidences else 1.0

        return {
            "text": final_text,
            "confidence": avg_confidence,
            "word_confidences": word_confidences
        }

    def _convert_character(self, class_name: str, capitalize: bool, number_mode: bool) -> str:
        """Convert a single Braille character to text"""

        # Handle special cases
        if class_name == "dot_4":
            return ""

        # Handle ñ/Ñ (important for Filipino)
        if class_name == "enye" or class_name == "ENYE":
            return "Ñ" if capitalize or class_name == "ENYE" else "ñ"

        # Handle numbers
        if number_mode and class_name.lower() in "abcdefghij":
            number_map = {
                'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5',
                'f': '6', 'g': '7', 'h': '8', 'i': '9', 'j': '0'
            }
            return number_map.get(class_name.lower(), class_name)

        # Handle already detected numbers
        if class_name in "0123456789":
            return class_name

        # Handle letters
        if len(class_name) == 1 and class_name.isalpha():
            if capitalize or class_name.isupper():
                return class_name.upper()
            else:
                return class_name.lower()

        return class_name

    def _correct_line(self, text: str, confidence: float) -> str:
        """Apply bilingual spell checking and corrections"""
        if not self.spell_checker:
            return text

        words = text.split()
        corrected_words = []

        for word in words:
            # Skip short words, numbers, and punctuation
            if len(word) <= 2 or word.isdigit() or not word.replace('ñ', 'n').replace('Ñ', 'N').isalpha():
                corrected_words.append(word)
                continue

            # Check spelling in both languages
            is_correct, correction, lang = self.spell_checker.check(word)

            if not is_correct and correction and correction != word.lower():
                # Calculate similarity
                similarity = SequenceMatcher(
                    None, word.lower(), correction).ratio()

                # Only apply if correction is similar enough
                if similarity > 0.6:
                    # Preserve original capitalization
                    if word[0].isupper():
                        correction = correction.capitalize()

                    lang_name = "English" if lang == 'en' else "Filipino" if lang == 'tl' else "Unknown"

                    self.corrections.append(CorrectionInfo(
                        original=word,
                        corrected=correction,
                        confidence=confidence,
                        method=f'spellcheck_{lang}'
                    ))

                    corrected_words.append(correction)
                    print(
                        f"✓ Na-correct ({lang_name}): '{word}' → '{correction}' (similarity: {similarity:.2f})")
                else:
                    corrected_words.append(word)
                    print(
                        f"⚠️  Hindi sigurado: '{word}' (suggested: '{correction}', pero mababang similarity: {similarity:.2f})")
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)

    def get_correction_suggestions(self, word: str) -> Dict[str, List[str]]:
        """Get alternative spelling suggestions in both languages"""
        if not self.spell_checker:
            return {}

        return self.spell_checker.get_candidates(word)

    def get_detection_report(self, results) -> Dict:
        """Get comprehensive report including corrections and confidence"""
        metadata = self.convert_results_to_text(results, return_metadata=True)

        report = {
            "final_text": metadata['text'],
            "raw_text": metadata.get('raw_text', ''),
            "total_detections": metadata['num_detections'],
            "num_lines": metadata['num_lines'],
            "average_confidence": metadata['average_confidence'],
            "corrections_made": len(metadata['corrections']),
            "llm_corrections_made": len(metadata.get('llm_corrections', [])),
            "corrections": [
                {
                    "original": c.original,
                    "corrected": c.corrected,
                    "method": c.method,
                    "confidence": c.confidence,
                    "language": "Filipino" if c.method.endswith('_tl') else "English" if c.method.endswith('_en') else "Unknown"
                }
                for c in metadata['corrections']
            ],
            "llm_corrections": metadata.get('llm_corrections', []),
            "low_confidence_words": metadata['low_confidence_words'],
            "quality_score": self._calculate_quality_score(metadata)
        }

        return report

    def _calculate_quality_score(self, metadata: Dict) -> float:
        """Calculate overall quality score for the detection"""
        conf_score = metadata['average_confidence']
        correction_penalty = len(metadata['corrections']) * 0.05
        low_conf_penalty = len(metadata['low_confidence_words']) * 0.03
        quality = max(0.0, conf_score - correction_penalty - low_conf_penalty)
        return quality

    def suggest_parameter_adjustments(self, results) -> Dict[str, str]:
        """Analyze results and suggest parameter adjustments"""
        detections = self._extract_detections(results[0])

        if not detections:
            return {"message": "Walang nakitang detection"}

        suggestions = {}

        # Analyze detection confidences
        confidences = [d.confidence for d in detections]
        avg_conf = np.mean(confidences)
        min_conf = np.min(confidences)

        if avg_conf < 0.4:
            suggestions['confidence'] = f"Mababang average confidence ({avg_conf:.2f}). Subukan:\n" \
                "  - Mas magandang ilaw\n" \
                "  - Mas mataas na resolution\n" \
                "  - I-retrain ang model gamit ang mas maraming data"

        if min_conf < 0.15:
            suggestions['threshold'] = f"Napakababang minimum confidence ({min_conf:.2f}). " \
                f"Itaas ang --conf threshold sa {min_conf + 0.1:.2f}"

        # Analyze spacing
        lines = self._organize_into_lines(detections)
        for line in lines:
            if len(line) > 1:
                gaps = [line[i+1].center_x -
                        line[i].center_x for i in range(len(line)-1)]
                gap_std = np.std(gaps)

                if gap_std > 20:
                    suggestions['spacing'] = f"Hindi consistent ang spacing ng characters (std: {gap_std:.1f}). Subukan:\n" \
                        "  - I-adjust ang --char-gap threshold\n" \
                        "  - Ayusin ang alignment ng image"

        return suggestions if suggestions else {"message": "Mukhang okay ang parameters!"}

    def add_custom_filipino_words(self, words: List[str]):
        """Add custom Filipino words to the dictionary"""
        if self.spell_checker and self.spell_checker.tl_checker:
            self.spell_checker.tl_checker.word_frequency.load_words(words)
            print(f"✓ Naidagdag ang {len(words)} custom Filipino words")
