"""
Document Tally Workflow - Main Application
Clean, modular version of the integrated compliance app
"""

import streamlit as st
import os
from datetime import datetime

# Import our modules
from factfind_extractor import FactFindExtractor, get_pdf_libraries, get_doc_libraries
from document_tally import (
    load_checklist_items, 
    generate_direct_document_tally,
    get_supported_transcript_formats,
    extract_and_format_excel_transcript,
    extract_and_format_docx_transcript, 
    extract_and_format_doc_transcript,
    parse_factfind_json
)
from llm_analyzer import LLMAnalyzer
from ui_components import (
    save_extraction_results,
    display_extraction_results,
    display_tally_results,
    display_complete_compliance_results
)

# Configuration
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "/home/work/IntageAudit/classification/model/Qwen2.5-32B-Instruct")

# App configuration
st.set_page_config(
    page_title="Document Tally Workflow",
    layout="wide"
)


def main():
    st.title("Document Tally Workflow")
    st.markdown("**Document Tally Analysis System**")
    
    # Initialize components
    llm_analyzer = LLMAnalyzer(VLLM_API_BASE, VLLM_MODEL)
    
    # Check system status
    if not check_system_status(llm_analyzer):
        st.stop()
    
    # Load checklist items
    checklist_items = load_checklist_items()
    if not checklist_items:
        st.error("Cannot load compliance checklist items.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("Analysis Workflow")
        st.subheader("Features")
        st.info("""
        **Complete compliance analysis workflow:**
        1. Extract fact-find data from PDF
        2. Upload transcript Excel file
        3. Generate comprehensive compliance report
        4. Download complete analysis results
        """)
    
    # Main interface
    show_complete_workflow_interface(checklist_items, llm_analyzer)


def check_system_status(llm_analyzer: LLMAnalyzer) -> bool:
    """Check if all required systems are available"""
    
    # Check vLLM connection
    if not llm_analyzer.test_connection():
        st.error("Cannot connect to vLLM server. Please ensure it's running.")
        return False
    
    # Check PDF libraries
    pdf_libs = get_pdf_libraries()
    if not pdf_libs:
        st.error("No PDF processing libraries available. Please install PyMuPDF or PyPDF2.")
        return False
    
    # Check document processing libraries
    doc_libs = get_doc_libraries()
    if not doc_libs:
        st.warning("Document processing not available. Install 'unstructured' package for DOCX/DOC support.")
    
    return True


def show_complete_workflow_interface(checklist_items: dict, llm_analyzer: LLMAnalyzer):
    """Show interface for complete compliance analysis workflow"""
    st.header("Document Tally Workflow")
    
    # Step 1: Fact-Find Extraction
    with st.expander("Step 1: Extract Fact-Find Data from Document", expanded=True):
        handle_factfind_extraction(llm_analyzer)
    
    # Step 2: Document Tally Analysis
    if st.session_state.get('pdf_processed', False):
        with st.expander("Step 2: Direct Document Tally Analysis", expanded=True):
            handle_document_tally_analysis(checklist_items, llm_analyzer)


def handle_factfind_extraction(llm_analyzer: LLMAnalyzer):
    """Handle fact-find document extraction"""
    
    # Determine supported file types
    supported_types = ["pdf"]
    type_descriptions = ["PDF documents"]
    
    doc_libs = get_doc_libraries()
    if "DOC" in doc_libs:
        supported_types.append("doc")
        type_descriptions.append("Word DOC documents")
    if "DOCX" in doc_libs:
        supported_types.append("docx")
        type_descriptions.append("Word DOCX documents")
    
    uploaded_file = st.file_uploader(
        "Upload Insurance Fact-Find Document",
        type=supported_types,
        help=f"Supported formats: {', '.join(type_descriptions)}"
    )
    
    if uploaded_file:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        st.success(f"{file_extension.upper()} uploaded: {uploaded_file.name}")
        
        if st.button("Extract Fact-Find Data", type="primary"):
            extractor = FactFindExtractor(VLLM_API_BASE, VLLM_MODEL)
            
            with st.spinner(f"Extracting data from {file_extension.upper()}..."):
                # Extract text based on file type
                raw_text = extract_text_from_file(extractor, uploaded_file, file_extension)
                
                if not raw_text.strip():
                    st.error(f"Could not extract text from {file_extension.upper()} document")
                    return
                
                # Extract with LLM
                extracted_data = extractor.extract_with_llm(raw_text)
            
            st.success("Fact-find data extracted successfully!")
            
            # Save extraction results and store in session state
            save_extraction_results(extracted_data, raw_text)
            st.session_state['extracted_factfind'] = extracted_data
            st.session_state['pdf_processed'] = True
            
            # Display results
            display_extraction_results(extractor, extracted_data, raw_text)
            
            st.info("**Extraction Summary:** Data extracted and ready for compliance analysis")


def extract_text_from_file(extractor: FactFindExtractor, uploaded_file, file_extension: str) -> str:
    """Extract text from uploaded file based on extension"""
    
    if file_extension == ".pdf":
        return extractor.extract_text_from_pdf(uploaded_file)
    
    elif file_extension == ".doc":
        return extractor.extract_text_from_doc(uploaded_file)
    
    elif file_extension == ".docx":
        # Handle DOCX files
        temp_docx_path = f"temp_docx_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        with open(temp_docx_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        try:
            from unstructured.partition.docx import partition_docx
            from unstructured.documents.elements import Text, Title, NarrativeText
            
            elements = partition_docx(temp_docx_path)
            raw_text = "\n".join([el.text for el in elements if isinstance(el, (Text, Title, NarrativeText))])
            formatted_text = f"--- Document Content ---\n{raw_text}"
            return formatted_text
        finally:
            if os.path.exists(temp_docx_path):
                os.remove(temp_docx_path)
    
    else:
        st.error(f"Unsupported file format: {file_extension}")
        return ""


def handle_document_tally_analysis(checklist_items: dict, llm_analyzer: LLMAnalyzer):
    """Handle document tally analysis"""
    
    st.success("Fact-find data ready from Step 1")
    
    # Get supported transcript formats
    supported_formats, format_descriptions = get_supported_transcript_formats()
    
    transcript_file = st.file_uploader(
        "Upload Transcript File",
        type=supported_formats,
        help=f"Supported formats: {', '.join(format_descriptions)}",
        key="complete_workflow_transcript"
    )
    
    if transcript_file:
        st.success(f"Transcript uploaded: {transcript_file.name}")
        
        if st.button("Generate Complete Compliance Report", type="primary"):
            # Use the extracted fact-find data from session state
            factfind_data = st.session_state['extracted_factfind']
            
            with st.spinner("Generating comprehensive compliance report..."):
                # Process transcript file
                transcript_content = process_transcript_file(transcript_file)
                
                if not transcript_content.strip():
                    st.error("Transcript file appears to be empty after processing")
                    return
                
                # Generate document tally
                tally_df = generate_direct_document_tally(
                    transcript_content, 
                    factfind_data, 
                    checklist_items, 
                    llm_analyzer
                )
                
                if tally_df.empty:
                    st.error("Failed to generate document tally")
                    return
                
                st.success("Complete compliance analysis generated successfully!")
                
                # Display comprehensive results
                display_complete_compliance_results(tally_df, transcript_file.name)
    else:
        st.info("Upload the transcript file to complete the analysis")


def process_transcript_file(transcript_file) -> str:
    """Process transcript file based on format"""
    
    file_extension = os.path.splitext(transcript_file.name)[1].lower()
    temp_transcript = f"temp_{transcript_file.name}"
    
    with open(temp_transcript, 'wb') as f:
        f.write(transcript_file.getbuffer())
    
    try:
        if file_extension == '.docx':
            transcript_content = extract_and_format_docx_transcript(temp_transcript)
        elif file_extension == '.doc':
            transcript_content = extract_and_format_doc_transcript(temp_transcript)
        elif file_extension in ['.xlsx', '.xls']:
            transcript_content = extract_and_format_excel_transcript(temp_transcript)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return ""
        
        return transcript_content
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_transcript):
            os.remove(temp_transcript)


if __name__ == "__main__":
    main()