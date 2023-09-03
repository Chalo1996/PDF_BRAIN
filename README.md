# PDFAI

**Setting Up the Environment**

1. Craete a virtual environment
```
a. If you prefer venv:
python3.11 -m venv <env_name>

b. If you prefer conda:
conda create --name <env_name> python=3.11

```
2. Activate the virtual environment
```
a. Using venv:
source <env_name>/bin/activate

b. Using conda:
conda activate <env_name>
```
**Install the modules**
```
python3 -m install -r requirements.txt
NB: Have pip installed.
```
**Required Keys**
```
OPENAI_API_KEY=XXXXXXXXXXXX
```
**Run the Application**
```
streamlit run app.py
```
***Upload a document and chat.***
