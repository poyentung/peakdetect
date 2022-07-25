from torch.utils.data import Dataset
from peakdetect.utils.dp_generator import DPGenerator

import hydra
from omegaconf import DictConfig
from peakdetect.utils import hydra_logging

log = hydra_logging.get_logger(__name__)

def data_generation(config:DictConfig):
    log.info(f"Instantiating dp_generator <{config._target_}>")
    dp_generator: Dataset = hydra.utils.instantiate(config)

    dp_generator.create_datasets()

@hydra.main(version_base=None, config_path="peakdetect/configs/", config_name="data.yaml")
def main(config: DictConfig):

    # Applies optional utilities
    hydra_logging.extras(config)

    # Train model
    data_generation(config)


if __name__ == "__main__":
    main()