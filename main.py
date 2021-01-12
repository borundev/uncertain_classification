import os

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from models.resnet import CustomModel
from models.simple_cnn import Net
from data import CustomDataModule
import names

def run_with_mod(mod, name=None, max_epochs=5):
    #pl.seed_everything(42)
    wandb_logger = WandbLogger(name=name,
                               project='uncertain_classification__fixed_extras')
    trainer = pl.Trainer(
        gpus=0,
        max_epochs=max_epochs,
        #limit_train_batches=50,
        #limit_val_batches=10,
        logger=wandb_logger,
        )
    model = CustomModel()
    cdm = CustomDataModule(mod=mod)
    trainer.fit(model, cdm)
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model_extra_samples_resnet_34_{'
                                                               '}.pt'.format(mod - 1)))
    wandb.finish()
    return model, cdm, trainer

mod=2
for _ in range(5):
    run_with_mod(mod,'resnet_34_fixed_{}_{}'.format(mod-1,names.get_full_name().replace(' ','_').lower()))
