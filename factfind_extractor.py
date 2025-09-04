"""
Fact-Find Data Extraction Module
Handles PDF/DOC/DOCX text extraction and LLM-based data extraction
"""

import streamlit as st
import pandas as pd
import json
import os
import re
import io
from datetime import datetime
from typing import Dict, List, Any, Optional
import requests

# PDF processing imports
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# DOCX/DOC processing imports
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


class FactFindExtractor:
    def __init__(self, vllm_api_base: str, vllm_model: str):
        self.vllm_api_base = vllm_api_base
        self.vllm_model = vllm_model
        self.raw_text = ""
        self.extracted_data = {}
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF using available libraries"""
        text = ""
        
        if PYMUPDF_AVAILABLE:
            try:
                pdf_bytes = pdf_file.read()
                pdf_file.seek(0)
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    text += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
                doc.close()
                return text
            except Exception as e:
                st.error(f"PyMuPDF error: {str(e)}")
        
        if PYPDF2_AVAILABLE:
            try:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"--- Page {i + 1} ---\n{page_text}\n\n"
                return text
            except Exception as e:
                st.error(f"PyPDF2 error: {str(e)}")
        
        return text
    
    def extract_text_from_doc(self, doc_file) -> str:
        """Extract text from DOC file using unstructured"""
        try:
            # Save file temporarily for processing
            temp_doc_path = f"temp_doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.doc"
            with open(temp_doc_path, 'wb') as f:
                f.write(doc_file.read())
            
            # Extract using unstructured
            elements = partition_doc(temp_doc_path)
            raw_text = "\n".join([el.text for el in elements if isinstance(el, (Text, Title, NarrativeText))])
            
            # Clean up temp file
            if os.path.exists(temp_doc_path):
                os.remove(temp_doc_path)
            
            # Parse for conversation structure like DOCX
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
            
            return "\n".join(formatted_lines) if formatted_lines else raw_text
        
        except Exception as e:
            st.error(f"Error processing DOC file: {str(e)}")
            return ""
    
    def _estimate_tokens(self, text: str) -> int:
        """Token estimation (1 token ≈ 3.5 characters - conservative for safety)"""
        return int(len(text) / 3.5)
    
    def _chunk_pdf_text(self, pdf_text: str, max_tokens_per_chunk: int = 8000) -> List[str]:
        """Intelligently chunk PDF text by pages while staying within token limits"""
        pages = pdf_text.split('--- Page ')
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for i, page in enumerate(pages):
            if i == 0:
                continue  # Skip empty first split
            
            page_text = f"--- Page {page}"
            page_tokens = self._estimate_tokens(page_text)
            
            # If single page exceeds limit, truncate it
            if page_tokens > max_tokens_per_chunk:
                page_text = page_text[:int(max_tokens_per_chunk * 3.5)]  # 3.5 chars per token
                page_tokens = max_tokens_per_chunk
            
            # Check if adding this page would exceed the limit
            if current_tokens + page_tokens > max_tokens_per_chunk and current_chunk:
                chunks.append(current_chunk)
                current_chunk = page_text
                current_tokens = page_tokens
            else:
                current_chunk += "\n" + page_text if current_chunk else page_text
                current_tokens += page_tokens
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def call_vllm_api(self, messages: List[Dict[str, str]], max_tokens: int = 2000, temperature: float = 0.1) -> str:
        """Call vLLM API with messages"""
        try:
            payload = {
                "model": self.vllm_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            response = requests.post(
                f"{self.vllm_api_base}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "bypass-tunnel-reminder": "true",
                    "User-Agent": "StreamlitApp/1.0"
                },
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                st.error(f"vLLM API error: {response.status_code} - {response.text}")
                return ""
        except Exception as e:
            st.error(f"Error calling vLLM: {str(e)}")
            return ""
    
    def extract_with_llm(self, pdf_text: str) -> Dict[str, Any]:
        """Use LLM to extract structured data from fact-find form with intelligent chunking"""
        
        # Check if text is within token limits
        estimated_tokens = self._estimate_tokens(pdf_text)
        
        if estimated_tokens <= 8000:  # Single extraction
            return self._extract_single_chunk(pdf_text)
        else:  # Multi-chunk extraction
            return self._extract_multi_chunk(pdf_text)
    
    def _extract_single_chunk(self, pdf_text: str) -> Dict[str, Any]:
        """Extract from single chunk of PDF text"""
        
        system_prompt = """Extract structured data from insurance fact-find PDF text.

**KEY RULES:**
- PDF numbers may concatenate: "7,000.007,000.00" = two 7,000.00 values
- Checkbox "Yes ● No" means "No" is selected
- is_documented=true if field has data, false if blank
- Use "--- Page X ---" markers for page_number
- Extract PRIMARY/MAIN values when multiple exist

**Output JSON Structure:** Return valid JSON with all fields having "value"/"level"/"status" etc, "is_documented" (boolean), "page_number" (integer):
{"personal_particulars":{"age_dob":{"age":null,"date_of_birth":null,"is_documented":false,"page_number":null},"education":{"level":null,"is_documented":false,"page_number":null},"language_proficiency":{"value":null,"is_documented":false,"page_number":null},"contact_info":{"mobile":null,"email":null,"is_documented":false,"page_number":null}},"cka_information":{"financial_education":{"value":null,"is_documented":false,"page_number":null},"work_experience":{"value":null,"is_documented":false,"page_number":null},"investment_experience":{"value":null,"is_documented":false,"page_number":null}},"financial_information":{"financial_objectives":{"goals":null,"time_horizon":null,"budget":null,"is_documented":false,"page_number":null},"risk_profile":{"value":null,"is_documented":false,"page_number":null},"employment":{"status":null,"occupation":null,"is_documented":false,"page_number":null},"income":{"annual_income":null,"monthly_income":null,"is_documented":false,"page_number":null},"assets":{"cash_equivalents":null,"bank_deposits":null,"total_assets":null,"is_documented":false,"page_number":null},"liabilities":{"total_liabilities":null,"loans":null,"is_documented":false,"page_number":null},"expenses":{"general_expenses":null,"insurance_premiums":null,"other_expenses":null,"is_documented":false,"page_number":null},"current_portfolio":{"existing_policies":null,"is_documented":false,"page_number":null},"dependents":{"number":null,"support_requirements":null,"is_documented":false,"page_number":null}}}

**PAGE NUMBER RULES:**
- Look for "--- Page X ---" markers in the text  
- When you find a FORM SECTION (even if blank), use the page number from where that section appears
- The content AFTER "--- Page X ---" belongs to page X
- NEVER set page_number to null - always find the page where the form section exists
- If a field has no value but the form section exists, still record the page number

**IMPORTANT:**
- Asset/Liability tables exist in most fact-find forms - find them even if values are blank
- Work/Investment experience often in questionnaire format ("Q1. How many years...")
- Risk profiling has specific questions and final assessment
- Look for both filled values AND blank form fields"""

        user_prompt = f"""Extract structured data from this insurance fact-find document:

{pdf_text}

Follow the exact JSON format specified. Return only the JSON structure, no additional text."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.call_vllm_api(messages, max_tokens=2000, temperature=0.1)
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                # Apply validation and correction
                result = self.validate_and_correct_extraction(result)
                return result
            else:
                # Try parsing entire response
                result = json.loads(response)
                result = self.validate_and_correct_extraction(result)
                return result
                
        except Exception as e:
            st.error(f"JSON parsing failed: {str(e)}")
            st.error(f"Raw LLM response: {response[:500]}...")
            return self.create_empty_structure()
    
    def _extract_multi_chunk(self, pdf_text: str) -> Dict[str, Any]:
        """Extract from multiple chunks and merge results"""
        
        chunks = self._chunk_pdf_text(pdf_text)
        st.info(f"Large document detected. Processing in {len(chunks)} chunks...")
        
        # Create base structure
        final_result = self.create_empty_structure()
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            st.write(f"Processing chunk {i+1}/{len(chunks)}...")
            
            # Extract from this chunk
            chunk_result = self._extract_single_chunk(chunk)
            
            # Merge non-empty results into final result
            final_result = self._merge_extraction_results(final_result, chunk_result)
        
        return final_result
    
    def _merge_extraction_results(self, base_result: Dict[str, Any], chunk_result: Dict[str, Any]) -> Dict[str, Any]:
        """Merge extraction results, prioritizing non-null values"""
        
        def merge_field_group(base_group: Dict[str, Any], chunk_group: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in chunk_group.items():
                if key in ["is_documented", "page_number"]:
                    # For boolean/page fields, update if chunk has better info
                    if value is not None and (base_group.get(key) is None or not base_group.get(key)):
                        base_group[key] = value
                elif value is not None and str(value).strip() != "":
                    # For content fields, update if chunk has value and base doesn't
                    if base_group.get(key) is None or str(base_group.get(key)).strip() == "":
                        base_group[key] = value
            return base_group
        
        # Merge each category
        for category_name, category_data in chunk_result.items():
            if isinstance(category_data, dict):
                if category_name not in base_result:
                    base_result[category_name] = {}
                
                for field_group_name, field_group_data in category_data.items():
                    if isinstance(field_group_data, dict):
                        if field_group_name not in base_result[category_name]:
                            base_result[category_name][field_group_name] = {}
                        
                        base_result[category_name][field_group_name] = merge_field_group(
                            base_result[category_name][field_group_name],
                            field_group_data
                        )
        
        return base_result
    
    def validate_and_correct_extraction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply robust validation and correction logic to extracted data"""
        try:
            # Financial consistency validation
            if "financial_information" in data:
                fin_info = data["financial_information"]
                
                # Assets validation: Check for parent-child inconsistencies
                if "assets" in fin_info and isinstance(fin_info["assets"], dict):
                    assets = fin_info["assets"]
                    total = self._parse_numeric_value(assets.get("total_assets"))
                    cash = self._parse_numeric_value(assets.get("cash_equivalents"))
                    bank = self._parse_numeric_value(assets.get("bank_deposits"))
                    
                    # PDF jumbling fix: If total = cash + bank AND cash = bank, likely double counting
                    if total and cash and bank and cash == bank and total == (cash + bank):
                        assets["total_assets"] = str(cash)  # Use the component value as the correct total
                    elif total and cash and total == cash and bank and bank == cash:
                        pass  # All three values are identical - categorization scenario
                    elif total and cash and bank and total == (cash + bank) and cash != bank:
                        pass  # Different component values that legitimately sum - keep as is
                    elif not total and cash:
                        if bank and bank != cash:
                            assets["total_assets"] = str(max(cash, bank))
                        else:
                            assets["total_assets"] = str(cash)
            
            # Documentation status consistency
            for category in data.values():
                if isinstance(category, dict):
                    for field_group in category.values():
                        if isinstance(field_group, dict):
                            # If any field has a value, mark as documented
                            has_values = False
                            for key, value in field_group.items():
                                if key not in ["is_documented", "page_number"] and value is not None and str(value).strip() != "":
                                    has_values = True
                                    break
                            
                            # Update documentation status based on actual values
                            if has_values and not field_group.get("is_documented", False):
                                field_group["is_documented"] = True
                            elif not has_values and field_group.get("is_documented", True):
                                field_group["is_documented"] = False
            
            return data
            
        except Exception as e:
            return data
    
    def _parse_numeric_value(self, value) -> float:
        """Safely parse numeric values from various formats"""
        if not value:
            return None
        try:
            clean_value = str(value).replace(",", "").replace("$", "").replace("SGD", "").replace(" ", "")
            numeric_match = re.search(r'[\d,]+\.?\d*', clean_value)
            if numeric_match:
                return float(numeric_match.group(0).replace(",", ""))
            return None
        except:
            return None
    
    def create_empty_structure(self) -> Dict[str, Any]:
        """Create empty structure if extraction fails"""
        return {
            "personal_particulars": {
                "age_dob": {"age": None, "date_of_birth": None, "is_documented": False, "page_number": None},
                "education": {"level": None, "is_documented": False, "page_number": None},
                "language_proficiency": {"value": None, "is_documented": False, "page_number": None},
                "contact_info": {"mobile": None, "email": None, "is_documented": False, "page_number": None}
            },
            "cka_information": {
                "financial_education": {"value": None, "is_documented": False, "page_number": None},
                "work_experience": {"value": None, "is_documented": False, "page_number": None},
                "investment_experience": {"value": None, "is_documented": False, "page_number": None}
            },
            "financial_information": {
                "financial_objectives": {"goals": None, "time_horizon": None, "budget": None, "is_documented": False, "page_number": None},
                "risk_profile": {"value": None, "is_documented": False, "page_number": None},
                "employment": {"status": None, "occupation": None, "is_documented": False, "page_number": None},
                "income": {"annual_income": None, "monthly_income": None, "is_documented": False, "page_number": None},
                "assets": {"cash_equivalents": None, "bank_deposits": None, "total_assets": None, "is_documented": False, "page_number": None},
                "liabilities": {"total_liabilities": None, "loans": None, "is_documented": False, "page_number": None},
                "expenses": {"general_expenses": None, "insurance_premiums": None, "other_expenses": None, "is_documented": False, "page_number": None},
                "current_portfolio": {"existing_policies": None, "is_documented": False, "page_number": None},
                "dependents": {"number": None, "support_requirements": None, "is_documented": False, "page_number": None}
            }
        }
    
    def convert_to_excel(self, extracted_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert extracted data to Excel format with individual item IDs for each sub-field"""
        rows = []
        
        # Map individual sub-fields to their specific item IDs
        field_to_item_mapping = {
            # Personal Particulars
            ("personal_particulars", "age_dob", "age"): "item_002",
            ("personal_particulars", "age_dob", "date_of_birth"): "item_002", 
            ("personal_particulars", "education", "level"): "item_003",
            ("personal_particulars", "language_proficiency", "value"): "item_004",
            ("personal_particulars", "contact_info", "mobile"): "item_005",
            ("personal_particulars", "contact_info", "email"): "item_005",
            
            # CKA Information
            ("cka_information", "financial_education", "value"): "item_007",
            ("cka_information", "work_experience", "value"): "item_008",
            ("cka_information", "investment_experience", "value"): "item_009",
            
            # Financial Information - split the merged ones
            ("financial_information", "financial_objectives", "goals"): "item_012",
            ("financial_information", "financial_objectives", "time_horizon"): "item_013",
            ("financial_information", "financial_objectives", "budget"): "item_014",
            ("financial_information", "risk_profile", "value"): "item_016",
            ("financial_information", "employment", "occupation"): "item_018",
            ("financial_information", "employment", "status"): "item_018",
            ("financial_information", "income", "annual_income"): "item_022",
            ("financial_information", "income", "monthly_income"): "item_022",
            ("financial_information", "assets", "total_assets"): "item_020",
            ("financial_information", "assets", "cash_equivalents"): "item_020",
            ("financial_information", "assets", "bank_deposits"): "item_020",
            ("financial_information", "liabilities", "total_liabilities"): "item_021",
            ("financial_information", "liabilities", "loans"): "item_021",
            ("financial_information", "expenses", "general_expenses"): "item_023",
            ("financial_information", "expenses", "insurance_premiums"): "item_023",
            ("financial_information", "expenses", "other_expenses"): "item_023",
            ("financial_information", "current_portfolio", "existing_policies"): "item_025",
            ("financial_information", "dependents", "number"): "item_029",
            ("financial_information", "dependents", "support_requirements"): "item_030"
        }
        
        # Define the correct regulatory order for processing
        regulatory_order = [
            # Personal Particulars
            ("personal_particulars", "age_dob", "age"),
            ("personal_particulars", "age_dob", "date_of_birth"),
            ("personal_particulars", "education", "level"),
            ("personal_particulars", "language_proficiency", "value"),
            ("personal_particulars", "contact_info", "mobile"),
            ("personal_particulars", "contact_info", "email"),
            
            # CKA Information
            ("cka_information", "financial_education", "value"),
            ("cka_information", "work_experience", "value"),
            ("cka_information", "investment_experience", "value"),
            
            # Financial Information - in correct regulatory order
            ("financial_information", "financial_objectives", "goals"),
            ("financial_information", "financial_objectives", "time_horizon"),
            ("financial_information", "financial_objectives", "budget"),
            ("financial_information", "risk_profile", "value"),
            ("financial_information", "employment", "occupation"),
            ("financial_information", "employment", "status"),
            ("financial_information", "assets", "total_assets"),
            ("financial_information", "assets", "cash_equivalents"),
            ("financial_information", "assets", "bank_deposits"),
            ("financial_information", "liabilities", "total_liabilities"),
            ("financial_information", "liabilities", "loans"),
            ("financial_information", "income", "annual_income"),
            ("financial_information", "income", "monthly_income"),
            ("financial_information", "expenses", "general_expenses"),
            ("financial_information", "expenses", "insurance_premiums"),
            ("financial_information", "expenses", "other_expenses"),
            ("financial_information", "current_portfolio", "existing_policies"),
            ("financial_information", "dependents", "number"),
            ("financial_information", "dependents", "support_requirements")
        ]
        
        # Process fields in the correct regulatory order
        for category_key, field_group_key, field_key in regulatory_order:
            # Check if this field exists in the extracted data
            if (category_key in extracted_data and 
                field_group_key in extracted_data[category_key] and 
                field_key in extracted_data[category_key][field_group_key]):
                
                field_group_data = extracted_data[category_key][field_group_key]
                field_value = field_group_data[field_key]
                page_number = field_group_data.get("page_number")
                
                # Handle page number properly
                if page_number is None or page_number == "null" or page_number == "":
                    page_number = "Not found"
                else:
                    page_number = str(page_number)
                
                # Get specific item ID for this field
                lookup_key = (category_key, field_group_key, field_key)
                item_id = field_to_item_mapping.get(lookup_key, "UNMAPPED")
                
                # Determine if this specific field is documented
                field_documented = field_value is not None and str(field_value).strip() != ""
                
                rows.append({
                    "Item_ID": str(item_id),
                    "Category": category_key.replace("_", " ").title(),
                    "Field_Group": field_group_key.replace("_", " ").title(),
                    "Field_Name": field_key.replace("_", " ").title(),
                    "Value": str(field_value) if field_value is not None else "NOT FOUND",
                    "Documentation_Status": "DOCUMENTED" if field_documented else "NOT DOCUMENTED",
                    "Page_Number": str(page_number)  # Ensure string type
                })
        
        return pd.DataFrame(rows)


# Library availability check functions
def get_pdf_libraries():
    """Get available PDF processing libraries"""
    available_libs = []
    if PYMUPDF_AVAILABLE:
        available_libs.append("PyMuPDF")
    if PYPDF2_AVAILABLE:
        available_libs.append("PyPDF2")
    return available_libs

def get_doc_libraries():
    """Get available document processing libraries"""
    doc_libs = []
    if DOCX_AVAILABLE:
        doc_libs.append("DOCX")
    if DOC_AVAILABLE:
        doc_libs.append("DOC")
    return doc_libs