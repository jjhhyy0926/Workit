import traceback
from rag.inference import load_model
try:
    load_model()
    print("OK")
except Exception:
    traceback.print_exc()
