"""
UI Components Module
Handles all Streamlit UI components and display functions
"""

import streamlit as st
import pandas as pd
import json
import io
import os
from datetime import datetime
from typing import Dict, Any


def save_extraction_results(extracted_data: Dict[str, Any], raw_text: str, temp_dir: str = None):
    """Save extraction results to temp directory"""
    import tempfile
    
    if temp_dir is None:
        # Use system temp directory or create local tmp folder
        try:
            temp_dir = tempfile.mkdtemp(prefix="documenttally_")
        except:
            # Fallback to current directory tmp folder
            temp_dir = os.path.join(os.getcwd(), "tmp")
    
    os.makedirs(temp_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON
    json_file = f"{temp_dir}/extraction_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(extracted_data, f, indent=2)
    
    # Save raw text
    text_file = f"{temp_dir}/raw_text_{timestamp}.txt"
    with open(text_file, 'w') as f:
        f.write(raw_text)
    
    st.info(f"Results saved to: {temp_dir}")
    st.success(f"Files: extraction_{timestamp}.json, raw_text_{timestamp}.txt")


def display_extraction_results(extractor, extracted_data: Dict[str, Any], raw_text: str):
    """Display extraction results in tabs"""
    tab1, tab2, tab3, tab4 = st.tabs(["Structured Data", "Excel View", "Downloads", "Raw Text"])
    
    with tab1:
        st.subheader("Extracted Structured Data")
        
        # Summary metrics
        def count_documented_fields(data, count={"total": 0, "documented": 0}):
            for key, value in data.items():
                if isinstance(value, dict):
                    if "is_documented" in value:
                        count["total"] += 1
                        if value["is_documented"]:
                            count["documented"] += 1
                    else:
                        count_documented_fields(value, count)
            return count
        
        counts = count_documented_fields(extracted_data)
        completion_rate = (counts["documented"] / counts["total"]) * 100 if counts["total"] > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fields Documented", counts["documented"])
        with col2:
            st.metric("Total Fields", counts["total"])
        with col3:
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
        
        # Display by category
        for category, category_data in extracted_data.items():
            with st.expander(f"{category.replace('_', ' ').title()}"):
                for field_group, field_data in category_data.items():
                    is_documented = field_data.get("is_documented", False)
                    status_icon = "✅" if is_documented else "❌"
                    
                    st.write(f"{status_icon} **{field_group.replace('_', ' ').title()}**")
                    
                    for key, value in field_data.items():
                        if key != "is_documented":
                            if value is not None:
                                st.write(f"   • {key.replace('_', ' ').title()}: `{value}`")
                            else:
                                st.write(f"   • {key.replace('_', ' ').title()}: *Not found*")
    
    with tab2:
        st.subheader("Excel Format View")
        excel_df = extractor.convert_to_excel(extracted_data)
        st.dataframe(excel_df, width='stretch')
    
    with tab3:
        st.subheader("Download Extracted Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Excel download
            excel_df = extractor.convert_to_excel(extracted_data)
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                excel_df.to_excel(writer, index=False, sheet_name='FactFind_Data')
            excel_buffer.seek(0)
            
            st.download_button(
                label="Download Excel (.xlsx)",
                data=excel_buffer.getvalue(),
                file_name=f"FactFind_Extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            # JSON download
            json_str = json.dumps(extracted_data, indent=2)
            
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"FactFind_Extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Preview JSON structure
        st.subheader("JSON Structure Preview")
        st.json(extracted_data)
    
    with tab4:
        st.subheader("Raw Extracted Text")
        st.text_area("Document Text Content", raw_text, height=400)


def display_tally_results(tally_df: pd.DataFrame, transcript_filename: str, factfind_filename: str):
    """Display document tally results"""
    # Display summary metrics
    st.subheader("Compliance Summary")
    code_counts = tally_df['Code'].value_counts()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Code 1", code_counts.get("Code 1", 0), help="Not mentioned but documented")
    with col2:
        st.metric("Code 2", code_counts.get("Code 2", 0), help="Mentioned but not documented")
    with col3:
        st.metric("Code 3", code_counts.get("Code 3", 0), help="Mentioned but incorrect")
    with col4:
        st.metric("Code 4", code_counts.get("Code 4", 0), help="Perfect match")
    with col5:
        total_items = len(tally_df[tally_df['Code'] != 'N/A'])
        compliance_rate = (code_counts.get("Code 4", 0) / total_items * 100) if total_items > 0 else 0
        st.metric("Compliance Rate", f"{compliance_rate:.1f}%", help="Percentage of Code 4 items")
    
    # Display results table
    st.subheader("Document Tally Results")
    
    # Filter options
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        code_filter = st.multiselect(
            "Filter by Code",
            options=["Code 1", "Code 2", "Code 3", "Code 4", "N/A"],
            default=["Code 1", "Code 2", "Code 3", "Code 4"]
        )
    
    with filter_col2:
        confidence_threshold = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Filter by LLM confidence score"
        )
    
    # Apply filters
    filtered_df = tally_df[
        (tally_df['Code'].isin(code_filter)) & 
        (tally_df['Confidence'].astype(float) >= confidence_threshold)
    ]
    
    st.dataframe(filtered_df, width='stretch')
    
    # Download section
    st.subheader("Download Results")
    
    # Create Excel report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Ensure tmp directory exists
    import tempfile
    try:
        tmp_dir = tempfile.mkdtemp(prefix="documenttally_results_")
    except:
        tmp_dir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
    
    # Define file paths
    excel_filepath = os.path.join(tmp_dir, f"document_tally_{timestamp}.xlsx")
    json_filepath = os.path.join(tmp_dir, f"document_tally_{timestamp}.json")
    
    # Use BytesIO to create Excel in memory
    excel_buffer = io.BytesIO()
    
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        # Main results
        tally_df.to_excel(writer, sheet_name='Document_Tally', index=False)
        
        # Code summary
        code_summary = pd.DataFrame({
            'Code': ['Code 1', 'Code 2', 'Code 3', 'Code 4'],
            'Description': [
                'Not mentioned but documented',
                'Mentioned but not documented',
                'Mentioned but incorrect',
                'Perfect match'
            ],
            'Count': [
                code_counts.get("Code 1", 0),
                code_counts.get("Code 2", 0),
                code_counts.get("Code 3", 0),
                code_counts.get("Code 4", 0)
            ]
        })
        code_summary.to_excel(writer, sheet_name='Code_Summary', index=False)
        
        # File sources
        sources_df = pd.DataFrame({
            'Source': ['Transcript File', 'Fact-Find File', 'Generated'],
            'Filename': [transcript_filename, factfind_filename, f'document_tally_{timestamp}.xlsx'],
            'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * 3
        })
        sources_df.to_excel(writer, sheet_name='Sources', index=False)
    
    excel_buffer.seek(0)
    
    # Save Excel file to tmp directory
    with open(excel_filepath, 'wb') as f:
        f.write(excel_buffer.getvalue())
    
    # Save JSON file to tmp directory
    json_data = tally_df.to_json(orient='records', indent=2)
    with open(json_filepath, 'w') as f:
        f.write(json_data)
    
    # Show saved file paths
    st.success(f"Results saved to:")
    st.code(excel_filepath)
    st.code(json_filepath)
    
    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Excel Report",
            data=excel_buffer.getvalue(),
            file_name=f"document_tally_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"document_tally_{timestamp}.json",
            mime="application/json"
        )
    
    # Code explanation
    with st.expander("Compliance Code Legend"):
        st.markdown("""
        **Code 1**: Not mentioned in transcript but was documented in fact-find
        - *Issue*: Documentation without conversation evidence
        - *Action*: Verify if documentation is accurate or if conversation was missed
        
        **Code 2**: Mentioned in transcript but left blank in fact-find form  
        - *Issue*: Missing documentation despite conversation
        - *Action*: Complete fact-find form with discussed information
        
        **Code 3**: Mentioned in transcript but documented incorrectly in fact-find
        - *Issue*: Incorrect or contradictory documentation
        - *Action*: Correct fact-find form to match transcript discussion
        
        **Code 4**: Mentioned in transcript and documented correctly in fact-find
        - *Status*: Perfect compliance match
        - *Action*: No action needed - excellent compliance
        
        **N/A**: Item not in compliance checklist or mapping
        - *Status*: Outside scope of current compliance requirements
        """)


def display_complete_compliance_results(tally_df: pd.DataFrame, transcript_filename: str):
    """Display results from complete workflow"""
    st.success("Complete Compliance Analysis Finished!")
    
    # Display enhanced summary
    st.subheader("Comprehensive Compliance Summary")
    
    code_counts = tally_df['Code'].value_counts()
    total_items = len(tally_df[tally_df['Code'] != 'N/A'])
    compliance_rate = (code_counts.get("Code 4", 0) / total_items * 100) if total_items > 0 else 0
    
    # Enhanced metrics display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Items Analyzed", total_items)
        st.metric("Perfect Compliance (Code 4)", code_counts.get("Code 4", 0))
        
    with col2:
        st.metric("Overall Compliance Rate", f"{compliance_rate:.1f}%")
        st.metric("Issues Found (Code 1-3)", 
                 code_counts.get("Code 1", 0) + code_counts.get("Code 2", 0) + code_counts.get("Code 3", 0))
    
    with col3:
        st.metric("Missing Documentation (Code 2)", code_counts.get("Code 2", 0))
        st.metric("Incorrect Documentation (Code 3)", code_counts.get("Code 3", 0))
    
    # Detailed breakdown
    st.subheader("Detailed Compliance Breakdown")
    
    breakdown_col1, breakdown_col2 = st.columns(2)
    
    with breakdown_col1:
        # Code distribution chart data
        code_data = {
            "Code 1 (Not mentioned but documented)": code_counts.get("Code 1", 0),
            "Code 2 (Mentioned but not documented)": code_counts.get("Code 2", 0),
            "Code 3 (Mentioned but incorrect)": code_counts.get("Code 3", 0),
            "Code 4 (Perfect compliance)": code_counts.get("Code 4", 0)
        }
        
        st.bar_chart(code_data)
    
    with breakdown_col2:
        # Priority actions
        st.subheader("Priority Actions Required")
        
        if code_counts.get("Code 3", 0) > 0:
            st.error(f"**HIGH PRIORITY**: {code_counts.get('Code 3', 0)} items with incorrect documentation")
        
        if code_counts.get("Code 2", 0) > 0:
            st.warning(f"**MEDIUM PRIORITY**: {code_counts.get('Code 2', 0)} items missing documentation")
        
        if code_counts.get("Code 1", 0) > 0:
            st.info(f"**REVIEW NEEDED**: {code_counts.get('Code 1', 0)} items documented without clear evidence")
        
        if compliance_rate >= 90:
            st.success("**EXCELLENT COMPLIANCE** - Above 90% compliance rate!")
        elif compliance_rate >= 75:
            st.success("**GOOD COMPLIANCE** - Above 75% compliance rate")
        elif compliance_rate >= 50:
            st.warning("**FAIR COMPLIANCE** - Above 50% compliance rate")
        else:
            st.error("**POOR COMPLIANCE** - Below 50% compliance rate")
    
    # Display full results table
    display_tally_results(tally_df, transcript_filename, "Extracted from document")