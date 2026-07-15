import torch

from unified_forgery_model import UnifiedForgeryModel


class cross_main(UnifiedForgeryModel):

    def __init__(self):
        super().__init__()
        self.set_training_stage("stage3")

    def forward(self, image):
        return super().forward(image, compute_loss=False)

    @torch.no_grad()
    def predict(self, image):
        self.eval()
        return self.forward(image)

