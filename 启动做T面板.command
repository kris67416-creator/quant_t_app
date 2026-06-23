#!/bin/bash
cd "$(dirname "$0")"
pip install -r requirements.txt --quiet 2>/dev/null
streamlit run live_signal.py --browser.gatherUsageStats false
