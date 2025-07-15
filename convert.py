import os
import subprocess

# List of your notebook files
notebooks = [
    "Production.ipynb", "Gasoline.ipynb", "Diesel.ipynb",
    "HeatingOil.ipynb", "JetFuel.ipynb", "Imports.ipynb",
    "Exports.ipynb", "Refinery.ipynb", "SandD.ipynb", "run.ipynb"
]

for nb in notebooks:
    base = os.path.splitext(nb)[0]
    py_file = f"{base}.py"
    subprocess.run(["jupyter", "nbconvert", "--to", "script", nb])
    print(f"✅ Converted {nb} → {py_file}")
