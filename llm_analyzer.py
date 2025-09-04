"""
LLM Analysis Module
Handles all LLM-based analysis for compliance checking
"""

import streamlit as st
import json
import requests
from typing import Dict, List, Any
from openai import OpenAI
import httpx


class LLMAnalyzer:
    def __init__(self, vllm_api_base: str, vllm_model: str):
        self.vllm_api_base = vllm_api_base
        self.vllm_model = vllm_model
    
    def test_connection(self):
        """Test connection to vLLM server"""
        try:
            response = requests.get(f"{self.vllm_api_base}/models", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
    
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
    
    def analyze_transcript_vs_factfind(self, item_id, question, full_transcript, factfind_evidence, factfind_values):
        """Use LLM to directly analyze transcript content against factfind documentation"""
        
        # Check transcript size and chunk if necessary
        estimated_tokens = len(full_transcript) // 4
        
        if estimated_tokens >= 12000:
            return self._analyze_chunked_transcript(item_id, question, full_transcript, factfind_evidence, factfind_values)
        else:
            return self._analyze_single_transcript(item_id, question, full_transcript, factfind_evidence, factfind_values)
    
    def _analyze_single_transcript(self, item_id, question, transcript_content, factfind_evidence, factfind_values):
        """Analyze a single transcript chunk against factfind documentation"""
        try:
            # Use custom headers for tunnel bypass if needed
            httpx_client = httpx.Client(
                headers={
                    "bypass-tunnel-reminder": "true",
                    "User-Agent": "StreamlitApp/1.0"
                }
            )
            client = OpenAI(base_url=self.vllm_api_base, api_key="dummy-key", http_client=httpx_client)
            
            # Prepare factfind information
            factfind_info = "No documentation found"
            if factfind_evidence != "Not mapped to fact-find fields":
                factfind_info = factfind_evidence
                if factfind_values:
                    factfind_info += f" | Values: {factfind_values}"
            
            system_prompt = """You are an expert compliance analyst specializing in insurance sales documentation review. Your task is to analyze the complete sales conversation transcript and compare it directly with fact-find form documentation.

COMPLIANCE CODES:
- **Code 1**: Not mentioned in transcript but was documented in fact-find (Documentation without evidence)
- **Code 2**: Mentioned in transcript but left blank/missing in fact-find (Missing documentation)  
- **Code 3**: Mentioned in transcript but documented incorrectly in fact-find (Incorrect documentation)
- **Code 4**: Mentioned in transcript and documented correctly in fact-find (Perfect compliance)

ANALYSIS PRINCIPLES:
1. **Direct Transcript Analysis**: Search the ENTIRE transcript for relevant information, not just extracted evidence
2. **Semantic Equivalence**: Different words can convey the same meaning - focus on substance over syntax
3. **Mathematical Relationships**: Values may be expressed differently but remain mathematically consistent  
4. **Professional Terminology**: Industry terms and colloquial descriptions often refer to the same concept
5. **Temporal Logic**: Consider realistic time relationships and age calculations based on conversation context
6. **Contextual Accuracy**: Documentation is correct if it captures the essential information discussed
7. **Age/DOB Special Case**: For item_002 (Age/Date of Birth), if client's age is mentioned in conversation AND documented in fact-find (even if expressed differently), this should be Code 4. Age and birth year are semantically equivalent.
8. **Budget/Financial Planning**: For item_014 (Budget), conversations often include multiple financial amounts (initial payment, monthly comfort level, annual budget). If the documented amount is mathematically consistent with ANY discussed amount (e.g., monthly × 12 = annual), this is Code 4.
9. **Multi-Part Discussions**: Conversations may cover different aspects of the same topic. Documentation capturing the most relevant/sustainable amount discussed is correct.
10. **Comprehensive Search**: Look throughout the transcript for ANY mention of the topic, including casual references, examples, or implied information.

CRITICAL RULE: Only assign Code 3 when there is genuine contradiction in meaning or substance, not merely different expression of the same information. Mathematical consistency indicates correct documentation."""

            user_prompt = f"""DIRECT COMPLIANCE ANALYSIS:

**Item ID**: {item_id}
**Compliance Question**: {question}

**SALES TRANSCRIPT CONTENT**:
{transcript_content}

**FACT-FIND DOCUMENTATION**:
{factfind_info}

**ANALYSIS TASK**:
1. **Comprehensive Transcript Search**: Read through the ENTIRE transcript and identify ANY discussion, mention, or reference related to this compliance item
2. **Information Extraction**: Extract all relevant information, values, statements, or context from the transcript
3. **Documentation Comparison**: Compare what you found in the transcript with the fact-find documentation
4. **Semantic Analysis**: Determine if the information aligns semantically and mathematically
5. **Code Assignment**: Assign the appropriate compliance code based on your analysis

**SEARCH STRATEGY**:
- Look for direct mentions of the topic
- Search for related concepts, synonyms, or industry terms
- Identify numerical values, dates, or specific data points
- Consider casual mentions or examples that relate to the compliance item
- Check for implied information or context clues

**REASONING REQUIREMENTS**:
- Quote specific transcript excerpts that relate to the compliance item
- IMPORTANT: When extracting evidence quotes, also look for timestamp information in the same row like [TIMESTAMP: Start_Time: XX:XX:XX,XXX | End_Time: XX:XX:XX,XXX]
- Extract ONLY the actual time values and include them in the timestamps array
- Explain your semantic analysis of both sources
- For financial amounts: Check if values are mathematically related (monthly×12=annual, etc.)
- For personal information: Consider different ways the same information might be expressed
- Justify the compliance code assignment with clear reasoning
- Highlight any context clues or industry-specific interpretations

**OUTPUT FORMAT** (JSON only):
{{
  "transcript_mentioned": true/false,
  "factfind_documented": true/false,
  "transcript_excerpts": ["quote1", "quote2", "..."],
  "timestamps": ["XX:XX:XX,XXX-XX:XX:XX,XXX", "..."],
  "values_align": true/false/null,
  "compliance_code": "Code 1/2/3/4",
  "confidence_score": 0.0-1.0,
  "analysis_reasoning": "Detailed explanation of your comparison and code assignment including specific transcript quotes",
  "value_comparison": "Specific comparison of discussed vs documented values (if applicable)",
  "search_summary": "Summary of how you searched the transcript and what you found"
}}"""

            response = client.chat.completions.create(
                model=self.vllm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1024,
                timeout=120
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result = json.loads(result_text)
                return result
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_start = result_text.find('{')
                json_end = result_text.rfind('}')
                if json_start != -1 and json_end != -1:
                    potential_json = result_text[json_start : json_end + 1]
                    return json.loads(potential_json)
                else:
                    return {
                        "transcript_mentioned": False,
                        "factfind_documented": False,
                        "transcript_excerpts": [],
                        "timestamps": [],
                        "values_align": None,
                        "compliance_code": "Code 2",
                        "confidence_score": 0.0,
                        "analysis_reasoning": "JSON parsing failed",
                        "value_comparison": "Unable to parse LLM response",
                        "search_summary": "Analysis failed due to parsing error"
                    }
                    
        except Exception as e:
            return {
                "transcript_mentioned": False,
                "factfind_documented": False,
                "transcript_excerpts": [],
                "timestamps": [],
                "values_align": None,
                "compliance_code": "Code 2",
                "confidence_score": 0.0,
                "analysis_reasoning": f"Error during LLM analysis: {str(e)}",
                "value_comparison": "Analysis failed due to error",
                "search_summary": "Analysis failed due to technical error"
            }
    
    def _analyze_chunked_transcript(self, item_id, question, full_transcript, factfind_evidence, factfind_values):
        """Handle chunked transcript analysis for large transcripts"""
        try:
            lines = full_transcript.split('\n')
            chunks = []
            current_chunk = ""
            
            for line in lines:
                if len(current_chunk) + len(line) < 12000 * 3:  # 12k token chunks
                    current_chunk += line + '\n'
                else:
                    chunks.append(current_chunk)
                    current_chunk = line + '\n'
            chunks.append(current_chunk)
            
            st.info(f"Large transcript detected for {item_id}. Processing in {len(chunks)} chunks...")
            
            # Process each chunk and collect all results
            chunk_results = []
            evidence_found = False
            
            for i, chunk in enumerate(chunks):
                st.write(f"Processing chunk {i+1}/{len(chunks)} for {item_id}...")
                
                result = self._analyze_single_transcript(item_id, question, chunk, factfind_evidence, factfind_values)
                chunk_results.append(result)
                
                # Check if evidence was found in this chunk
                if result.get('transcript_mentioned', False):
                    evidence_found = True
                    confidence = result.get('confidence_score', 0.0)
                    st.success(f"Found evidence for {item_id} in chunk {i+1} (confidence: {confidence:.2f})")
            
            # Return the best result from all chunks
            if evidence_found:
                # Find the chunk result with highest confidence
                best_result = max(chunk_results, key=lambda x: x.get('confidence_score', 0.0))
                best_result['search_summary'] = f"Analyzed {len(chunks)} chunks, found evidence. Using best result with confidence {best_result.get('confidence_score', 0.0):.2f}"
                return best_result
            
            # If no evidence found in any chunk
            return {
                "transcript_mentioned": False,
                "factfind_documented": False,
                "transcript_excerpts": [],
                "timestamps": [],
                "values_align": None,
                "compliance_code": "Code 2",
                "confidence_score": 0.0,
                "analysis_reasoning": "No evidence found in any transcript chunk",
                "value_comparison": "No comparison available",
                "search_summary": f"Analyzed all {len(chunks)} chunks but found no relevant information"
            }
            
        except Exception as e:
            st.error(f"Error in chunked analysis for {item_id}: {str(e)}")
            return {
                "transcript_mentioned": False,
                "factfind_documented": False,
                "transcript_excerpts": [],
                "timestamps": [],
                "values_align": None,
                "compliance_code": "Code 2",
                "confidence_score": 0.0,
                "analysis_reasoning": f"Error during chunked analysis: {str(e)}",
                "value_comparison": "Analysis failed due to error",
                "search_summary": "Chunked analysis failed due to technical error"
            }