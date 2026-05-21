$PythonExe = "C:\Users\SAI KEERTHI\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if (Test-Path $PythonExe) {
    & $PythonExe -m streamlit run app.py --server.headless true --server.port 8501 --server.address 127.0.0.1
} else {
    python -m streamlit run app.py --server.headless true --server.port 8501 --server.address 127.0.0.1
}
