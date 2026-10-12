@echo off
title Loan Approval Predictor
cd /d "%~dp0"
echo Installing requirements (first time only)...
pip install -r requirements.txt
echo Starting the app... it will open in your browser.
python -m streamlit run app.py
pause
