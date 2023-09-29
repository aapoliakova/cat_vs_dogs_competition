import torch.nn as nn
import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchvision.models import resnet50, ResNet50_Weights
import wandb


class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=5):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, 1)

        trainer.logger.experiment.log({
            "examples": [wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                         for x, pred, y in zip(val_imgs, preds, self.val_labels)],
            "global_step": trainer.global_step
        })


class BaseClassifier(pl.LightningModule):
    def __init__(self, lr):
        super(BaseClassifier, self).__init__()
        self.save_hyperparameters()

        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.valid_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_acc = Accuracy(task="multiclass", num_classes=2)
        self.wandb_check = {'train/acc': [], 'valid/acc': [], 'train/step': [], 'valid/step': []}

        self.loss = nn.CrossEntropyLoss()
        self.lr = lr

        self.model = resnet50(ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(2048, 2)

    def forward(self, inputs):
        logits = self.model(inputs)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid/loss"}

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss = self.loss(logits, labels)
        self.train_acc(logits, labels)

        print(self.trainer.global_step, "Before")
        self.log('train/loss', loss, on_step=True)
        print(self.trainer.global_step, "After")
        self.log('train/acc', self.train_acc, on_step=True)

        self.wandb_check['train/acc'].append(self.train_acc)
        self.wandb_check['train/step'].append(self.trainer.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss = self.loss(logits, labels)
        self.valid_acc(logits, labels)

        self.log('valid/loss', loss, on_step=True)
        self.log('valid/acc', self.valid_acc, on_step=True)

        self.wandb_check['valid/acc'].append(self.train_acc)
        self.wandb_check['valid/step'].append(self.trainer.global_step)
