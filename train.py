from typing import List, Optional
from peakdetect.data_module import DPDataModule
from peakdetect.lightning_module import EDPeakDector
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers import LightningLoggerBase

from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)

import hydra
from omegaconf import DictConfig

from peakdetect.utils import hydra_logging

log = hydra_logging.get_logger(__name__)

def train(config: DictConfig):
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    data_module: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
    
    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger)
    
    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    hydra_logging.log_hyperparameters(
        config=config,
        model=model,
        datamodule=data_module,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    if config.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=data_module)

    # Make sure everything closed properly
    log.info("Finalizing!")
    hydra_logging.finish(
        config=config,
        model=model,
        datamodule=data_module,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run") and config.get("train"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

@hydra.main(version_base=None, config_path="peakdetect/configs/", config_name="train.yaml")
def main(config: DictConfig):

    # Applies optional utilities
    hydra_logging.extras(config)

    # Train model
    train(config)


if __name__ == "__main__":
    main()