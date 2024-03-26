import sys
from pathlib import Path

file=Path(__file__).resolve()
parent, root =file.parent, file.parent[1]

sys.path.append(str(root))

import typing as t
from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from demand_model import __version__ as _version
from demand_model.config.core import DATASET_DIR,TRAINED_MODEL_DIR,config


##  Pre-Pipeline Preparation

# combine datasets to one dataframe

def combine_dataset(dataframe: pd.DataFrame, date_var: str):

    new_train = pd.merge(train, meal_info, how = "left", on = "meal_id")
    latest_train = pd.merge(new_train, fulfilment_center_info, how = "left", on = "center_id")
    
    return df



def _load_raw_dataset(*,file_name:str) -> pd.DataFrame:
    dataframe=pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*,file_name:str) -> pd.DataFrame:
    dataframe=pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame = dataframe)

    return transformed

def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:

    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*,file_name:str) -> Pipeline:
    """ Load a persisted pipeline """

    file_path= TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)

    return trained_model

def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()