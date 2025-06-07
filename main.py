import os
import functions_framework
import google.generativeai as genai

# Configure Google Gemini API (assuming API_KEY is set as an environment variable in Jules/GCP)
# In a real setup, manage API keys securely (e.g., Secret Manager)
# For prototyping, os.getenv is common.
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 0.2,  # Keep responses focused and less creative
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 500,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize the generative model
model = genai.GenerativeModel(
    model_name="gemini-pro", # Using gemini-pro for text tasks
    generation_config=generation_config,
    safety_settings=safety_settings
)

@functions_framework.http
def generate_assessment_summary(request):
    """HTTP Cloud Function that processes nurse input and generates a structured assessment.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`.
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>
    """
    request_json = request.get_json(silent=True)
    if not request_json or 'nurse_notes' not in request_json:
        return 'Please provide "nurse_notes" in the request body.', 400

    nurse_notes = request_json['nurse_notes']

    # Our few-shot prompt structure
    prompt_parts = [
        # System Prompt (our core identity and instructions)
        "You are Nightingale AI, a highly specialized and experienced critical care nursing assistant. Your purpose is to help critical care nurses document patient assessments efficiently and accurately. When provided with a nurse's raw, often informal, assessment notes, your task is to synthesize and structure the information into a clear, concise, and professional nursing assessment summary. Organize the summary using distinct, relevant categories such as \"PATIENT OVERVIEW,\" \"REASON FOR ADMISSION/PRIMARY CONCERN,\" \"DIAGNOSIS/KEY FINDINGS,\" and \"CURRENT INTERVENTIONS/SUPPORT.\" Always maintain a professional, supportive, and precise tone. Do not provide medical advice, diagnosis, or treatment recommendations. Focus solely on structuring and summarizing the provided assessment data.",
        "\n\n",
        # Few-shot example 1 (User Input) - Using the user-provided detailed one
        "Nurse Input: John Doe 52 DNR patient of the medical group (cc2/hospitalists) presents with sob, cta showed pe, on heparin gtt, still on non rebreather",
        "\n\n",
        # Few-shot example 1 (Desired AI Output) - Corresponding to the above
        "Nightingale AI Output:\nPATIENT OVERVIEW: John Doe, 52 y.o., DNR status (Medical Group: CC2/Hospitalists).\nREASON FOR ADMISSION/PRIMARY CONCERN: Shortness of Breath (SOB).\nDIAGNOSIS/KEY FINDINGS: CTA confirmed Pulmonary Embolism (PE).\nCURRENT INTERVENTIONS/SUPPORT: Initiated on Heparin gtt. Currently requiring Non-rebreather mask for oxygen support.",
        "\n\n",
        # Generic example for robustness - Systemic assessment
        "Nurse Input: Patient is alert and oriented x3, follows commands. Pupils equal, round, reactive to light and accommodation. Airway patent, clear lung sounds bilaterally, no cough. Sinus rhythm, S1S2 audible, radial pulses 2+, capillary refill <3s. Skin warm and dry, no lesions. Abdomen soft, non-tender, bowel sounds present. Voiding clear yellow urine. Patient moves all extremities equally. No pain reported.",
        "\n\n",
        "Nightingale AI Output:\nPATIENT OVERVIEW: [Patient Name/Age if provided, otherwise general status]\nNEURO: Alert and oriented x3, follows commands. PERRLA.\nRESP: Airway patent, clear lung sounds bilaterally. No cough.\nCARD: Sinus rhythm, S1S2 audible. Radial pulses 2+, cap refill <3s.\nGI/GU: Abdomen soft, non-tender, bowel sounds present. Voiding clear yellow urine.\nSKIN: Warm, dry, intact, no lesions.\nMS/PAIN: Moves all extremities equally. No pain reported.",
        "\n\n",
        # Actual user input to be processed
        "Nurse Input: " + nurse_notes,
        "\n\n",
        "Nightingale AI Output:",
    ]

    try:
        response = model.generate_content(prompt_parts)
        # Ensure the response is valid and extract text
        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            generated_text = response.candidates[0].content.parts[0].text
            return generated_text, 200
        else:
            # Log the problematic response for debugging
            print(f"Unexpected response structure: {response}")
            return "Failed to generate assessment summary: No content in response or unexpected structure.", 500
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"An error occurred: {str(e)}", 500
