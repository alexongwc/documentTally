"""
Document Tally Analysis Module
Handles transcript processing and compliance analysis
"""

import streamlit as st
import pandas as pd
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

# Document processing imports
try:
    from unstructured.partition.docx import partition_docx
    from unstructured.partition.doc import partition_doc  
    from unstructured.documents.elements import Text, Title, NarrativeText
    DOCX_AVAILABLE = True
    DOC_AVAILABLE = True
except ImportError:
    try:
        from unstructured.partition.docx import partition_docx
        from unstructured.documents.elements import Text, Title, NarrativeText
        DOCX_AVAILABLE = True
        DOC_AVAILABLE = False
    except ImportError:
        DOCX_AVAILABLE = False
        DOC_AVAILABLE = False


def load_checklist_items(checklist_path: str = None):
    """Load checklist items from JSON file"""
    try:
        if checklist_path is None:
            checklist_path = os.path.join(os.path.dirname(__file__), "checklist_items.json")
            
        with open(checklist_path, 'r') as f:
            checklist_data = json.load(f)
        
        # Flatten checklist items
        flat_items = {}
        
        def flatten_items(items, parent_context=""):
            for item in items:
                current_question = f"{parent_context} > {item['question_text']}" if parent_context else item['question_text']
                
                if item.get('sub_items'):
                    flatten_items(item['sub_items'], current_question)
                else:
                    flat_items[item['id']] = {
                        'id': item['id'],
                        'question': current_question
                    }
        
        flatten_items(checklist_data)
        return flat_items
        
    except Exception as e:
        st.error(f"Error loading checklist items: {str(e)}")
        return {}


def extract_and_format_excel_transcript(excel_path):
    """Extract and format transcript from Excel file"""
    try:
        df = pd.read_excel(excel_path, sheet_name=None)
        formatted_lines = []
        
        for sheet_name, sheet_df in df.items():
            if sheet_df.empty:
                continue
                
            formatted_lines.append(f"=== SHEET: {sheet_name} ===")
            
            # Check for timestamp columns
            timestamp_cols = []
            for col in sheet_df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['time', 'timestamp', 'date', 'when']) or col_lower in ['at'] or col_lower.endswith('_at') or col_lower.startswith('at_'):
                    timestamp_cols.append(col)
            
            for index, row in sheet_df.iterrows():
                row_parts = []
                timestamp_info = ""
                
                # Extract timestamp information first if available
                if timestamp_cols:
                    timestamps = []
                    for ts_col in timestamp_cols:
                        if pd.notna(row[ts_col]) and str(row[ts_col]).strip() and str(row[ts_col]).strip().lower() not in ['', 'nan', 'none']:
                            timestamp_val = str(row[ts_col]).strip()
                            if timestamp_val and not timestamp_val.lower().startswith('status'):
                                timestamps.append(f"{ts_col}: {timestamp_val}")
                    if timestamps:
                        timestamp_info = f"[TIMESTAMP: {' | '.join(timestamps)}] "
                
                # Format all other columns (excluding timestamp columns from the main data)
                for col, val in row.items():
                    if pd.notna(val) and col not in timestamp_cols:
                        row_parts.append(f"{col}: {str(val)}")
                
                if row_parts:
                    formatted_lines.append(f"Row {index + 1}: {timestamp_info}{' | '.join(row_parts)}")
        
        return "\n".join(formatted_lines)
    except Exception as e:
        st.error(f"Error processing Excel transcript: {str(e)}")
        return ""


def extract_and_format_docx_transcript(docx_path):
    """Extract and format transcript from DOCX file"""
    try:
        elements = partition_docx(docx_path)
        raw_text = "\n".join([el.text for el in elements if isinstance(el, (Text, Title, NarrativeText))])
        parts = re.split(r'(MS:|A:)', raw_text)
        formatted_lines = []
        current_speaker = None
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            if part in ['MS:', 'A:']:
                current_speaker = part[:-1]
            elif current_speaker and part:
                text = part.strip()
                if text:
                    formatted_lines.append(f"[{current_speaker}]: {text}")
        return "\n".join(formatted_lines)
    except Exception as e:
        st.error(f"Error processing DOCX transcript: {str(e)}")
        return ""


def extract_and_format_doc_transcript(doc_path):
    """Extract and format transcript from DOC file"""
    try:
        elements = partition_doc(doc_path)
        raw_text = "\n".join([el.text for el in elements if isinstance(el, (Text, Title, NarrativeText))])
        parts = re.split(r'(MS:|A:)', raw_text)
        formatted_lines = []
        current_speaker = None
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            if part in ['MS:', 'A:']:
                current_speaker = part[:-1]
            elif current_speaker and part:
                text = part.strip()
                if text:
                    formatted_lines.append(f"[{current_speaker}]: {text}")
        return "\n".join(formatted_lines)
    except Exception as e:
        st.error(f"Error processing DOC transcript: {str(e)}")
        return ""


def parse_factfind_json(uploaded_file):
    """Parse fact-find results from JSON file"""
    try:
        factfind_data = json.load(uploaded_file)
        return factfind_data
    except Exception as e:
        st.error(f"Error parsing fact-find file: {str(e)}")
        return None


def generate_direct_document_tally(full_transcript, factfind_data, checklist_items, llm_analyzer):
    """Generate document tally by directly analyzing transcript vs factfind"""
    try:
        # Mapping between checklist items and factfind fields
        ITEM_TO_FACTFIND_MAPPING = {
            "item_002": ["personal_particulars", "age_dob"],  # Age/DOB
            "item_003": ["personal_particulars", "education"],  # Education
            "item_004": ["personal_particulars", "language_proficiency"],  # Language
            "item_005": ["personal_particulars", "contact_info"],  # Contact info
            "item_007": ["cka_information", "financial_education"],  # Financial education
            "item_008": ["cka_information", "work_experience"],  # Work experience
            "item_009": ["cka_information", "investment_experience"],  # Investment experience
            "item_012": ["financial_information", "financial_objectives"],  # Financial goals
            "item_013": ["financial_information", "financial_objectives"],  # Time horizon
            "item_014": ["financial_information", "financial_objectives"],  # Budget
            "item_016": ["financial_information", "risk_profile"],  # Risk profiling
            "item_017": ["financial_information", "risk_profile"],  # Risk assessment
            "item_018": ["financial_information", "employment"],  # Employment
            "item_020": ["financial_information", "assets"],  # Assets
            "item_021": ["financial_information", "liabilities"],  # Liabilities
            "item_022": ["financial_information", "income"],  # Income
            "item_023": ["financial_information", "expenses"],  # Expenses
            "item_025": ["financial_information", "current_portfolio"],  # Investment holdings
            "item_026": ["financial_information", "current_portfolio"],  # Insurance policies
            "item_029": ["financial_information", "dependents"],  # Number of dependents
            "item_030": ["financial_information", "dependents"],  # Support requirements
        }
        
        tally_results = []
        
        # Get all unique item IDs from checklist and mapping, sort them
        all_item_ids = set(checklist_items.keys()) | set(ITEM_TO_FACTFIND_MAPPING.keys())
        sorted_item_ids = sorted(all_item_ids, key=lambda x: int(x.split('_')[1]))
        
        for item_id in sorted_item_ids:
            # Get question text
            question = checklist_items.get(item_id, {}).get('question', f'Unknown question for {item_id}')
            
            # Skip items with unknown questions
            if question.startswith('Unknown question for'):
                tally_results.append({
                    "ItemID": item_id,
                    "Question": question,
                    "Evidence_from_Transcript": "N/A - Unknown compliance item",
                    "Evidence_from_FactFind": "N/A - Unknown compliance item", 
                    "LLM_Reasoning": "Item not in compliance checklist - no coding required",
                    "Value_Comparison": "N/A",
                    "Code": "N/A"
                })
                continue
            
            # Get factfind data and prepare for LLM analysis
            evidence_from_factfind = "Not mapped to fact-find fields"
            factfind_values = ""
            
            if item_id in ITEM_TO_FACTFIND_MAPPING:
                path = ITEM_TO_FACTFIND_MAPPING[item_id]
                
                # Navigate through factfind data structure
                current_data = factfind_data
                for key in path:
                    if isinstance(current_data, dict) and key in current_data:
                        current_data = current_data[key]
                    else:
                        current_data = None
                        break
                
                if current_data and isinstance(current_data, dict):
                    factfind_documented = current_data.get("is_documented", False)
                    page_num = current_data.get("page_number", "Unknown")
                    
                    if factfind_documented:
                        # Extract actual values for LLM analysis
                        values = []
                        for key, value in current_data.items():
                            if key not in ["is_documented", "page_number"] and value is not None:
                                values.append(f"{key}: {value}")
                        
                        if values:
                            evidence_from_factfind = f"Page {page_num}: Form section documented"
                            factfind_values = '; '.join(values)
                        else:
                            evidence_from_factfind = f"Page {page_num}: Form section documented but no specific values found"
                    else:
                        evidence_from_factfind = f"Page {page_num}: Form section exists but left blank"
            
            # Use LLM to directly analyze transcript vs factfind
            st.info(f"Analyzing {item_id}: {question[:50]}...")
            
            llm_result = llm_analyzer.analyze_transcript_vs_factfind(
                item_id=item_id,
                question=question,
                full_transcript=full_transcript,
                factfind_evidence=evidence_from_factfind,
                factfind_values=factfind_values
            )
            
            code = llm_result.get("compliance_code", "Code 2")
            confidence = llm_result.get("confidence_score", 0.0)
            reasoning = llm_result.get("analysis_reasoning", "No reasoning provided")
            value_comparison = llm_result.get("value_comparison", "No comparison available")
            transcript_excerpts = llm_result.get("transcript_excerpts", [])
            timestamps = llm_result.get("timestamps", [])
            
            # Combine excerpts with timestamps for Evidence_from_Transcript column
            evidence_parts = []
            if transcript_excerpts and timestamps:
                # Pair excerpts with timestamps (if available)
                for i, excerpt in enumerate(transcript_excerpts):
                    if i < len(timestamps) and timestamps[i] != "No timestamps found":
                        evidence_parts.append(f"{excerpt} [Time: {timestamps[i]}]")
                    else:
                        evidence_parts.append(excerpt)
            elif transcript_excerpts:
                evidence_parts = transcript_excerpts
            
            excerpts_display = "; ".join(evidence_parts) if evidence_parts else "No relevant excerpts found"
            
            # Update factfind evidence with LLM insights if values exist
            if factfind_values:
                evidence_from_factfind += f" | Values: {factfind_values}"
            
            tally_results.append({
                "ItemID": item_id,
                "Question": question,
                "Evidence_from_Transcript": excerpts_display,
                "Evidence_from_FactFind": evidence_from_factfind,
                "LLM_Reasoning": reasoning,
                "Value_Comparison": value_comparison,
                "Code": code,
                "Confidence": f"{confidence:.2f}"
            })
        
        return pd.DataFrame(tally_results)
        
    except Exception as e:
        st.error(f"Error generating direct document tally: {str(e)}")
        return pd.DataFrame()


def get_supported_transcript_formats():
    """Get supported transcript formats based on available libraries"""
    supported_formats = ["xlsx", "xls"]
    format_descriptions = ["Excel files"]
    
    if DOC_AVAILABLE:
        supported_formats.append("doc")
        format_descriptions.append("Word DOC documents")
    
    if DOCX_AVAILABLE:
        supported_formats.append("docx")
        format_descriptions.append("Word DOCX documents")
    
    return supported_formats, format_descriptions