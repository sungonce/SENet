r""" Test code of Structural Embedding Network (SENet, CVPR 2023)"""
# written by Seongwon Lee (won4113@yonsei.ac.kr)

import torch
import core.checkpoint as checkpoint
from core.config import cfg
from test.test_model import test_model
from model.SENet_model import SENet

def setup_model():
    """Sets up a model for training or testing and log the results."""
    # Build the model
    print("=> creating SENet model")
    model = SENet(cfg.SENET)
    print(model)
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)

    return model

def __main__():
    """Test the model."""
    if cfg.TEST.WEIGHTS == "":
        print("no test weights exist!!")
    else:
        # Construct the model
        model = setup_model()
        # Load checkpoint
        checkpoint.load_checkpoint(cfg.TEST.WEIGHTS, model)
        test_model(model, cfg.TEST.DATA_DIR, cfg.TEST.DATASET_LIST, cfg.TEST.SCALE_LIST)
