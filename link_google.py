from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
from PyPDF2 import PdfReader

def authenticate_google_drive():
    # Replace with your OAuth 2.0 client ID and secret
    client_id = "934751949274-25alttv32qdan45hfcppi99dtrjg7ho9.apps.googleusercontent.com"
    client_secret = "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"

    # Set up the OAuth 2.0 flow
    flow = InstalledAppFlow.from_client_config(
        {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        },
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )

    # Run the OAuth 2.0 flow
    credentials = flow.run_local_server(port=0)
    return credentials

def read_pdf(file_content):
    pdf = PdfReader(file_content)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def load_pdfs_from_google_drive(folder_id):
    credentials = authenticate_google_drive()
    service = build("drive", "v3", credentials=credentials)

    # Get the list of files in the specified folder
    results = service.files().list(q=f"'{folder_id}' in parents", fields="files(id, name)").execute()
    files = results.get("files", [])

    documents = []
    for file in files:
        if file["name"].endswith(".pdf"):
            request = service.files().get_media(fileId=file["id"])
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            file_content.seek(0)
            text = read_pdf(file_content)
            documents.append(text)

    return documents
