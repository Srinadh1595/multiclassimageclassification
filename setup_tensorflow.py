import os
import subprocess
import sys
import venv
import webbrowser

def run_command(command):
    subprocess.run(command, shell=True, check=True)

def main():
    # Step 1: Check if Python 3.10 is installed
    try:
        run_command("py -3.10 --version")
    except subprocess.CalledProcessError:
        print("Python 3.10 is not installed. Please install Python 3.10 from https://www.python.org/downloads/release/python-3100/")
        webbrowser.open("https://www.python.org/downloads/release/python-3100/")
        return

    # Step 2: Create a virtual environment
    venv_dir = "tf_env"
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        venv.create(venv_dir, with_pip=True)

    # Step 3: Activate the virtual environment
    activate_script = os.path.join(venv_dir, "Scripts", "activate")
    if not os.path.exists(activate_script):
        print("Virtual environment activation script not found. Please check your setup.")
        return

    # Step 4: Upgrade pip
    run_command(f"{activate_script} && python -m pip install --upgrade pip")

    # Step 5: Install requirements
    run_command(f"{activate_script} && pip install -r requirements.txt")

    # Step 6: Run the training script
    run_command(f"{activate_script} && python train_model.py/train.py")

if __name__ == "__main__":
    main() 