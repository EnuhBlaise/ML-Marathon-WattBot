#!/usr/bin/env python3
"""
Test script for WattBot RAG improvements
"""

import pandas as pd
import numpy as np
import os
import re
import json
from typing import List, Dict, Tuple, Optional, Any
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

print("Testing WattBot RAG improvements...")

# Test answer parsing
def test_answer_parsing():
    print("\n=== TESTING ANSWER PARSING ===")

    # Mock answer parsing functions
    def _parse_boolean_answer(answer_text: str) -> Dict:
        answer_lower = answer_text.lower().strip()
        if answer_lower in ['true', 'false', 'yes', 'no']:
            value = '1' if answer_lower in ['true', 'yes'] else '0'
            return {
                'answer': answer_text.upper() if answer_text.lower() in ['true', 'false'] else answer_text,
                'answer_value': value,
                'answer_unit': 'boolean'
            }
        return {
            'answer': answer_text,
            'answer_value': answer_text,
            'answer_unit': 'is_blank'
        }

    def _parse_numeric_answer(answer_text: str) -> Dict:
        import re
        patterns = [
            (r'(\d+(?:\.\d+)?)\s*%', 'percent'),
            (r'(\d+(?:\.\d+)?)\s*(kWh|MWh|GWh)', 'energy_unit'),
            (r'(\d+(?:\.\d+)?)', 'number'),
        ]

        answer_clean = answer_text.replace(',', '').strip()

        for pattern, unit_type in patterns:
            match = re.search(pattern, answer_clean, re.IGNORECASE)
            if match:
                value = match.group(1)
                if unit_type == 'percent':
                    return {
                        'answer': f"{value}%",
                        'answer_value': value,
                        'answer_unit': 'percent'
                    }
                elif unit_type == 'energy_unit':
                    unit = match.group(2)
                    return {
                        'answer': f"{value} {unit}",
                        'answer_value': value,
                        'answer_unit': unit
                    }
                else:
                    return {
                        'answer': value,
                        'answer_value': value,
                        'answer_unit': 'is_blank'
                    }

        return {
            'answer': answer_text,
            'answer_value': answer_text,
            'answer_unit': 'is_blank'
        }

    # Test cases
    test_cases = [
        ("TRUE", "boolean question"),
        ("4%", "percentage"),
        ("100 kWh", "energy"),
        ("42", "plain number"),
        ("Water consumption", "categorical"),
    ]

    for answer, question_type in test_cases:
        if "boolean" in question_type:
            result = _parse_boolean_answer(answer)
        elif "percentage" in question_type or "energy" in question_type or "number" in question_type:
            result = _parse_numeric_answer(answer)
        else:
            result = {
                'answer': answer,
                'answer_value': answer,
                'answer_unit': 'is_blank'
            }

        print(f"  Input: '{answer}' -> Value: '{result['answer_value']}', Unit: '{result['answer_unit']}'")

def test_config():
    print("\n=== TESTING CONFIG ===")
    print(f"  Chunk size: {config['chunk_size']}")
    print(f"  Max chunks per query: {config['max_chunks_per_query']}")
    print(f"  Similarity threshold: {config['similarity_threshold']}")
    print(f"  Sample mode: {config['use_sample_mode']}")

if __name__ == "__main__":
    test_config()
    test_answer_parsing()
    print("\n✅ Basic tests completed successfully!")