import os

# List your converted script files here
files_to_wrap = [
    "Production.py", "Gasoline.py", "Diesel.py", "HeatingOil.py", "JetFuel.py",
    "Imports.py", "Exports.py", "Refinery.py", "SD.py", "run.py"
]

def wrap_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Skip if already wrapped
    if any("def main()" in line for line in lines):
        print(f"✅ Skipping {filename} (already wrapped)")
        return

    # Indent all code lines
    indented = ["    " + line if line.strip() else line for line in lines]

    # Add main wrapper
    wrapped = (
        ["def main():\n"] +
        indented +
        ["\n\nif __name__ == \"__main__\":\n    main()\n"]
    )

    with open(filename, 'w') as f:
        f.writelines(wrapped)

    print(f"✅ Wrapped {filename}")

# Run for all files
for file in files_to_wrap:
    if os.path.exists(file):
        wrap_file(file)
    else:
        print(f"❌ File not found: {file}")
