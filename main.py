import argparse
import json
import logging
import os
import sys

import torch
import torch.utils.data
import torch.utils.data.distributed
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

from lm.inference import ModelWrapper
import json
import argparse
from shutil import copyfile

def model_fn(model_dir):
    return ModelWrapper.load(Path(model_dir))

def input_fn(request_body, request_content_type):
    assert request_content_type == 'application/json'
    return json.loads(request_body)
    
def predict_fn(input_object, model):
    tokens_gen = model.generate_tokens(input_object['tokens'], tokens_to_generate=42, top_k=8)
    return model.sp_model.DecodePieces(tokens_gen)
    
def output_fn(prediction, response_content_type):
    assert request_content_type == 'application/json'
    return json.dumps({"tokens": prediction})
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args, _ = parser.parse_known_args()

    model = model_fn(args.train)    
    for f in ['params.json', 'sp.model', 'model.pt']:
        copyfile(os.path.join(args.train, f), os.path.join(args._model_dir, f))
        
    
    
             
    
    
