import streamlit as st

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

FOLDER_ID = '1PphA7iXH-_YMbLrZxB8wdtTLXcAyEuRh'
CREDENTIALS = 'ivynatal_tpu.json'
SCOPES = 

# Authenticate and construct the Google Drive API client
credentials = service_account.Credentials.from_service_account_info(
    st.secrets['service_account'],
    scopes=['https://www.googleapis.com/auth/drive']
)
drive_service = build('drive', 'v3', credentials=credentials)

query = f"'{FOLDER_ID}' in parents and trashed=false"
results = drive_service.files().list(q=query, fields="files(id, name)", supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
files = results.get('files', [])

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
st.write(str(files))
