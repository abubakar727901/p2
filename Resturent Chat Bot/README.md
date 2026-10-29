# Zaiqa Palace Restaurant Chat

## Run locally

1. Install the packages with `pip install -r requirements.txt`.
2. Set `OPENROUTER_API_KEY` in a local `.env` file.
3. Run `streamlit run app.py`.

The `.env` file is for local use only and should not be committed.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository with these files in the repository
   root: `app.py`, `embeddings.py`, `pdf_reader.py`, `rag.py`,
   `requirements.txt`, and `restaurant_menu.pdf`.
2. In Streamlit Community Cloud, set the app's main file path to `app.py`.
   Do not add a folder name to this path when the files are in the repository root.
3. Keep the dependency file named `requirements.txt` in the repository root.
   Do not add a line such as `rEPORT READER/requirements.txt` or
   `-r rEPORT READER/requirements.txt` inside it. A folder path with spaces can
   be split and parsed as an invalid package requirement.
4. In the app's **Settings → Secrets**, add:

   ```toml
   OPENROUTER_API_KEY = "your_openrouter_api_key"
   ```

   Optional model settings can also be added there:

   ```toml
   OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-3-small"
   OPENROUTER_LLM_MODEL = "openai/gpt-4o-mini"
   ```

The app reads settings from Streamlit Secrets when deployed and from environment
variables or `.env` when run locally. Keep API keys out of source control.
