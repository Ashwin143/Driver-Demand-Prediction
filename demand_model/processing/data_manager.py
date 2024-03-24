import sys
from pathlib import Path

file=Path(__file__).resolve()
parent, root =file.parent, file.parent[1]

sys.path.append(str(root))

import typing as t
from pathlib import Path
import joblib
import pandas as pd

from demand_model import __version__ as _version
from demand_model.config.core import DATASET_DIR,TRAINED_MODEL_DIR,config




