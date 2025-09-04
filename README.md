# Document Tally Workflow

A modular compliance analysis system for insurance documentation using AI-powered document processing.

## 🎯 Features

- **Fact-Find Extraction**: Extract structured data from PDF/DOC/DOCX insurance documents
- **Compliance Analysis**: Compare transcripts vs fact-find documentation
- **AI-Powered**: Uses LLM for intelligent document analysis
- **Modular Design**: Clean, maintainable code structure
- **Multiple Formats**: Supports PDF, Excel, Word documents

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

### Environment Variables

For cloud deployment, set:
```bash
VLLM_API_BASE=https://your-api-endpoint/v1
VLLM_MODEL=your-model-path
```

## 📁 Project Structure

```
Documenttally/
├── app.py                    # Main Streamlit application
├── factfind_extractor.py     # PDF/DOC extraction & processing
├── document_tally.py         # Transcript processing & compliance analysis  
├── llm_analyzer.py           # LLM communication & analysis
├── ui_components.py          # Streamlit UI components
├── checklist_items.json      # Compliance checklist
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🔧 Usage

1. **Upload Insurance Document**: PDF, DOC, or DOCX fact-find forms
2. **Extract Data**: AI extracts structured information
3. **Upload Transcript**: Excel or Word transcript files
4. **Generate Report**: Comprehensive compliance analysis
5. **Download Results**: Excel and JSON outputs

## 🌐 Deployment

### Streamlit Cloud

1. Fork this repository
2. Connect to Streamlit Cloud
3. Set environment variables for your LLM endpoint
4. Deploy!

### Local with GPU Backend

1. Start vLLM server locally
2. Use tunnel service (ngrok, LocalTunnel, etc.)
3. Set tunnel URL in environment variables

## 📋 Requirements

- Python 3.8+
- Streamlit
- pandas
- OpenAI client
- unstructured (for document processing)
- PyPDF2 or PyMuPDF

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for compliance analysis purposes.