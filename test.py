# written by Seongwon Lee (won4113@yonsei.ac.kr)

import core.config as config
import core.SENet_tester as SENet_tester

from core.config import cfg

def main():
    config.load_cfg_fom_args("test a SENet model.")
    cfg.NUM_GPUS=1
    cfg.freeze()
    SENet_tester.__main__()

if __name__ == "__main__":
    main()
