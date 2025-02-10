from fasthtml.common import *
import os

app = FastHTML(
    hdrs=(
        Script(src="https://cdn.tailwindcss.com"),
        Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css")
    )
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
def get():
    form = Form(
        Div(
            Input(
                type="file",
                name="document",
                accept=".txt,.pdf",
                cls="file-input file-input-bordered file-input-primary w-full max-w-xs"
            ),
            Input(
                type="text",
                name="website",
                placeholder="Enter website URL",
                cls="input input-bordered input-primary w-full max-w-xs mt-4"
            ),
            Button(
                "Submit",
                type="submit",
                cls="btn btn-primary mt-4"
            ),
            cls="card bg-base-100 shadow-xl p-6"
        ),
        hx_post="/submit",
        enctype="multipart/form-data",
        method="post"
    )
    
    return Title("Upload & Enter Website"), Main(
        H1("Upload File or Enter Website", cls="text-2xl mb-4"), 
        form, 
        cls="container mx-auto p-4"
    )

@app.post("/submit")
async def handle_submission(request):
    form = await request.form()
    document = form.get("document")
    website = form.get("website")
    
    if document:
        file_path = os.path.join(UPLOAD_FOLDER, document.filename)
        contents = await document.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        return P(f"File '{document.filename}' uploaded successfully!", cls="text-success")
    
    if website:
        return P(f"Website entered: {website}", cls="text-info")
    
    return P("No input provided. Please upload a file or enter a website.", cls="text-error")

serve()