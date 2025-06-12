import shutil
import os

cache_dir = os.path.expanduser("~/.cache/torch/hub")
if os.name == 'nt':  # Windows path fix
    cache_dir = os.path.expandvars(r"%USERPROFILE%\.cache\torch\hub")

if os.path.exists(cache_dir):
    print(f"Removing torch hub cache at: {cache_dir}")
    shutil.rmtree(cache_dir)
    print("Cache cleared successfully!")
else:
    print(f"No torch hub cache found at: {cache_dir}")
