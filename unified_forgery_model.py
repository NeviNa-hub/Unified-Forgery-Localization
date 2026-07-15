from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from forensic_backbone import ForensicBackbone
from cross_manipulation_heads import (
    DomainInvariantBranch,
    FeatureDecouplingLoss,
    FrequencyPriorBranch,
    ManipulationSensitiveBranch,
    NonSemanticContextAdapter,
    SemanticContextAdapter,
)
from decoderhead import Multiple
from grl import GradientReversalLayer


class UnifiedForgeryModel(nn.Module):
    feature_keys = ('third1', 'third2', 'third3', 'third', 'last1', 'last')

    def __init__(self,
                 depth=[5, 8, 20, 7],
                 embed_dim=[64, 128, 320, 512],
                 head_dim=64,
                 img_size=512,
                 s_blocks3=[8, 4, 2, 1],
                 s_blocks4=[2, 1],
                 mlp_ratio=4,
                 qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 pretrained_path=None,
                 grl_lambda=0.01,
                 sensitive_loss_weight=0.05,
                 invariant_loss_weight=0.15,
                 confusion_loss_weight=0.01,
                 decouple_loss_weight=0.01,
                 generative_main_weight=0.7,
                 dice_loss_weight=0.5,
                 stage3_generative_main_weight=0.6,
                 stage2_main_fusion_max=0.15,
                 stage3_main_fusion_max=0.25,
                 stage2_main_injection_start=0.35,
                 invariant_main_logit_bound=2.5,
                 sensitive_main_logit_bound=2.0,
                 stage1_sensitive_main_fusion_scale=0.08,
                 stage2_sensitive_main_fusion_scale=0.20,
                 stage3_sensitive_main_fusion_scale=0.30,
                 generative_sensitive_fusion_ratio=0.25,
                 stage3_frequency_scale=0.25,
                 generative_aux_dilation_kernel=17,
                 use_semantic_adapter=True,
                 use_nonsemantic_adapter=True):
        super(UnifiedForgeryModel, self).__init__()
        self.img_size = img_size
        self.stage_name = 'stage1'

        self.base_sensitive_loss_weight = sensitive_loss_weight
        self.base_invariant_loss_weight = invariant_loss_weight
        self.base_confusion_loss_weight = confusion_loss_weight
        self.base_decouple_loss_weight = decouple_loss_weight
        self.generative_main_weight = generative_main_weight
        self.base_generative_main_weight = generative_main_weight
        self.stage3_generative_main_weight = stage3_generative_main_weight
        self.current_generative_main_weight = generative_main_weight
        self.dice_loss_weight = dice_loss_weight
        self.freeze_low_level_encoder = True
        self.freeze_release_progress = 0.4
        self.current_stage_progress = 0.0
        self.stage2_main_fusion_max = stage2_main_fusion_max
        self.stage3_main_fusion_max = stage3_main_fusion_max
        self.stage2_main_injection_start = stage2_main_injection_start
        self.invariant_main_logit_bound = invariant_main_logit_bound
        self.sensitive_main_logit_bound = sensitive_main_logit_bound
        self.stage1_sensitive_main_fusion_scale = stage1_sensitive_main_fusion_scale
        self.stage2_sensitive_main_fusion_scale = stage2_sensitive_main_fusion_scale
        self.stage3_sensitive_main_fusion_scale = stage3_sensitive_main_fusion_scale
        self.generative_sensitive_fusion_ratio = max(0.0, min(float(generative_sensitive_fusion_ratio), 1.0))
        self.stage3_frequency_scale = stage3_frequency_scale
        self.generative_aux_dilation_kernel = generative_aux_dilation_kernel
        self.use_semantic_adapter = use_semantic_adapter
        self.use_nonsemantic_adapter = use_nonsemantic_adapter

        self.encoder_net = ForensicBackbone(
            layers=depth,
            embed_dim=embed_dim,
            img_size=img_size,
            s_blocks3=s_blocks3,
            s_blocks4=s_blocks4,
            head_dim=head_dim,
            drop_path_rate=0.2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            pretrained_path=pretrained_path,
        )

        self.lmu = Multiple(embed_dim=embed_dim[-1])

        self.BCE_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.domain_loss = nn.CrossEntropyLoss()
        self.feature_decouple_loss = FeatureDecouplingLoss()

        self.nonsemantic_shallow_adapter = NonSemanticContextAdapter(
            channels=embed_dim[2],
        )
        self.nonsemantic_mid_adapter = NonSemanticContextAdapter(
            channels=embed_dim[2],
        )
        self.semantic_context_adapter = SemanticContextAdapter(
            channels=embed_dim[-1],
        )

        self.sensitive_branch = ManipulationSensitiveBranch(
            shallow_channels=embed_dim[2],
            mid_channels=embed_dim[2],
            hidden_channels=64,
        )
        self.invariant_branch = DomainInvariantBranch(
            mid_channels=embed_dim[2],
            high_channels=embed_dim[-1],
            hidden_channels=64,
        )
        self.frequency_branch = FrequencyPriorBranch(out_channels=128)
        self.invariant_fuse = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.invariant_main_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        self.sensitive_main_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        # shared output uses base_logits directly
        self.traditional_aux_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        self.generative_aux_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self.grl = GradientReversalLayer(lambda_=grl_lambda)
        self.grl_scalar = GradientReversalLayer(lambda_=grl_lambda)
        self.confusion_head_global = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        self._init_new_modules()

        nn.init.constant_(self.traditional_aux_head[-1].weight, 0.0)
        nn.init.constant_(self.traditional_aux_head[-1].bias, 0.0)
        nn.init.constant_(self.sensitive_main_head[-1].weight, 0.0)
        nn.init.constant_(self.sensitive_main_head[-1].bias, 0.0)

        self.set_training_stage('stage1')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_new_modules(self):
        modules = [
            self.nonsemantic_shallow_adapter,
            self.nonsemantic_mid_adapter,
            self.semantic_context_adapter,
            self.sensitive_branch,
            self.invariant_branch,
            self.frequency_branch,
            self.invariant_fuse,
            self.invariant_main_head,
            self.sensitive_main_head,
            self.traditional_aux_head,
            self.generative_aux_head,
            self.confusion_head_global,
        ]
        for module in modules:
            module.apply(self._init_weights)

    def set_grl_lambda(self, lambda_):
        self.grl.set_lambda(lambda_)

    def _set_module_trainable(self, module, trainable):
        for param in module.parameters():
            param.requires_grad = trainable

    def _set_module_frozen_state(self, module, trainable):
        self._set_module_trainable(module, trainable)
        if trainable:
            module.train()
        else:
            module.eval()

    def _set_encoder_low_level_trainable(self, trainable):
        low_level_modules = [
            self.encoder_net.patch_embed1,
            self.encoder_net.patch_embed2,
            self.encoder_net.blocks1,
            self.encoder_net.blocks2,
            self.encoder_net.norm1,
            self.encoder_net.norm2,
        ]
        for module in low_level_modules:
            self._set_module_frozen_state(module, trainable)

    def train(self, mode: bool = True):
        super().train(mode)
        self._apply_stage_module_modes()
        return self

    def _apply_stage_module_modes(self):
        if self.stage_name == 'stage1':
            if self.use_nonsemantic_adapter:
                self.nonsemantic_shallow_adapter.train()
                self.nonsemantic_mid_adapter.train()
            else:
                self.nonsemantic_shallow_adapter.eval()
                self.nonsemantic_mid_adapter.eval()
            self.semantic_context_adapter.eval()
            self.sensitive_branch.train()
            self.traditional_aux_head.train()
            self.sensitive_main_head.train()
            self.invariant_branch.eval()
            self.frequency_branch.eval()
            self.invariant_fuse.eval()
            self.invariant_main_head.eval()
            self.generative_aux_head.eval()
            self.confusion_head_global.eval()
        elif self.stage_name == 'stage2':
            if self.use_nonsemantic_adapter:
                self.nonsemantic_shallow_adapter.train()
                self.nonsemantic_mid_adapter.train()
            else:
                self.nonsemantic_shallow_adapter.eval()
                self.nonsemantic_mid_adapter.eval()
            if self.use_semantic_adapter:
                self.semantic_context_adapter.train()
            else:
                self.semantic_context_adapter.eval()
            self.sensitive_branch.train()
            self.sensitive_main_head.train()
            self.invariant_branch.train()
            self.frequency_branch.train()
            self.invariant_fuse.train()
            self.invariant_main_head.train()
            self.traditional_aux_head.train()
            self.generative_aux_head.train()
            self.confusion_head_global.train()
        elif self.stage_name == 'stage3':
            if self.use_nonsemantic_adapter:
                self.nonsemantic_shallow_adapter.train()
                self.nonsemantic_mid_adapter.train()
            else:
                self.nonsemantic_shallow_adapter.eval()
                self.nonsemantic_mid_adapter.eval()
            if self.use_semantic_adapter:
                self.semantic_context_adapter.train()
            else:
                self.semantic_context_adapter.eval()
            self.sensitive_branch.train()
            self.sensitive_main_head.train()
            self.invariant_branch.train()
            self.frequency_branch.eval()
            self.invariant_fuse.train()
            self.invariant_main_head.train()
            self.traditional_aux_head.train()
            self.generative_aux_head.train()
            self.confusion_head_global.eval()

    def set_training_stage(self, stage_name):
        if stage_name not in ('stage1', 'stage2', 'stage3'):
            raise ValueError(f"Unsupported stage: {stage_name}")

        self.stage_name = stage_name

        if stage_name == 'stage1':
            self._set_encoder_low_level_trainable(True)
            self._set_module_frozen_state(self.nonsemantic_shallow_adapter, self.use_nonsemantic_adapter)
            self._set_module_frozen_state(self.nonsemantic_mid_adapter, self.use_nonsemantic_adapter)
            self._set_module_frozen_state(self.semantic_context_adapter, False)
            self._set_module_frozen_state(self.sensitive_branch, True)
            self._set_module_frozen_state(self.sensitive_main_head, True)
            self._set_module_frozen_state(self.invariant_branch, False)
            self._set_module_frozen_state(self.frequency_branch, False)
            self._set_module_frozen_state(self.invariant_fuse, False)
            self._set_module_frozen_state(self.invariant_main_head, False)
            self._set_module_frozen_state(self.traditional_aux_head, True)
            self._set_module_frozen_state(self.generative_aux_head, False)
            self._set_module_frozen_state(self.confusion_head_global, False)

            self.sensitive_loss_weight = 0.15
            self.invariant_loss_weight = 0.0
            self.confusion_loss_weight = 0.0
            self.decouple_loss_weight = 0.0
            self.current_generative_main_weight = 1.0

        elif stage_name == 'stage2':
            if self.freeze_low_level_encoder:
                self._set_encoder_low_level_trainable(False)
            else:
                self._set_encoder_low_level_trainable(True)
            self._set_module_frozen_state(self.nonsemantic_shallow_adapter, self.use_nonsemantic_adapter)
            self._set_module_frozen_state(self.nonsemantic_mid_adapter, self.use_nonsemantic_adapter)
            self._set_module_frozen_state(self.semantic_context_adapter, self.use_semantic_adapter)
            self._set_module_frozen_state(self.sensitive_branch, True)
            self._set_module_frozen_state(self.sensitive_main_head, True)
            self._set_module_frozen_state(self.invariant_branch, True)
            self._set_module_frozen_state(self.frequency_branch, True)
            self._set_module_frozen_state(self.invariant_fuse, True)
            self._set_module_frozen_state(self.invariant_main_head, True)
            self._set_module_frozen_state(self.traditional_aux_head, True)
            self._set_module_frozen_state(self.generative_aux_head, True)
            self._set_module_frozen_state(self.confusion_head_global, True)

            self.sensitive_loss_weight = self.base_sensitive_loss_weight
            self.invariant_loss_weight = self.base_invariant_loss_weight
            self.confusion_loss_weight = self.base_confusion_loss_weight
            self.decouple_loss_weight = self.base_decouple_loss_weight
            self.current_generative_main_weight = self.base_generative_main_weight

        elif stage_name == 'stage3':
            self._set_encoder_low_level_trainable(False)
            self._set_module_frozen_state(self.nonsemantic_shallow_adapter, self.use_nonsemantic_adapter)
            self._set_module_frozen_state(self.nonsemantic_mid_adapter, self.use_nonsemantic_adapter)
            self._set_module_frozen_state(self.semantic_context_adapter, self.use_semantic_adapter)
            self._set_module_frozen_state(self.sensitive_branch, True)
            self._set_module_frozen_state(self.sensitive_main_head, True)
            self._set_module_frozen_state(self.invariant_branch, True)
            self._set_module_frozen_state(self.frequency_branch, False)
            self._set_module_frozen_state(self.invariant_fuse, True)
            self._set_module_frozen_state(self.invariant_main_head, True)
            self._set_module_frozen_state(self.traditional_aux_head, True)
            self._set_module_frozen_state(self.generative_aux_head, True)
            self._set_module_frozen_state(self.confusion_head_global, False)

            self.sensitive_loss_weight = self.base_sensitive_loss_weight
            self.invariant_loss_weight = self.base_invariant_loss_weight
            self.confusion_loss_weight = 0.0
            self.decouple_loss_weight = self.base_decouple_loss_weight * 0.5
            self.current_generative_main_weight = self.stage3_generative_main_weight
        self._apply_stage_module_modes()

    def set_stage_progress(self, stage_name, progress):
        self.current_stage_progress = progress
        if stage_name != 'stage2':
            return
        should_unfreeze = progress >= self.freeze_release_progress
        self._set_encoder_low_level_trainable(should_unfreeze)

    def _is_stage2_main_injection_enabled(self):
        return self.stage_name != 'stage2' or self.current_stage_progress >= self.stage2_main_injection_start

    def _get_main_fusion_scale(self):
        if self.stage_name == 'stage2':
            if not self._is_stage2_main_injection_enabled():
                return 0.0
            # Only start injecting invariant residual after stage2 warmup.
            effective = (self.current_stage_progress - self.stage2_main_injection_start) / max(1.0 - self.stage2_main_injection_start, 1e-6)
            return self.stage2_main_fusion_max * min(max(effective, 0.0), 1.0)
        if self.stage_name == 'stage3':
            return self.stage3_main_fusion_max
        return 0.0

    def _get_frequency_injection_scale(self):
        if self.stage_name == 'stage2':
            return 1.0
        if self.stage_name == 'stage3':
            return self.stage3_frequency_scale
        return 0.0

    def _get_sensitive_main_fusion_scale(self, domain_label):
        if self.stage_name == 'stage1':
            return torch.full_like(
                domain_label.view(-1, 1, 1, 1).float(),
                self.stage1_sensitive_main_fusion_scale,
            )

        base_scale = 0.0
        if self.stage_name == 'stage2':
            base_scale = self.stage2_sensitive_main_fusion_scale
        elif self.stage_name == 'stage3':
            base_scale = self.stage3_sensitive_main_fusion_scale

        conventional_scale = torch.full_like(domain_label.view(-1, 1, 1, 1).float(), base_scale)
        generative_scale = torch.full_like(
            domain_label.view(-1, 1, 1, 1).float(),
            base_scale * self.generative_sensitive_fusion_ratio,
        )
        return torch.where(
            domain_label.view(-1, 1, 1, 1) == 0,
            conventional_scale,
            generative_scale,
        )

    def _get_decoder_features(self, encoder_outputs):
        missing = [key for key in self.feature_keys if key not in encoder_outputs]
        if missing:
            raise KeyError(f"Missing encoder outputs: {missing}")
        return {key: encoder_outputs[key] for key in self.feature_keys}

    def _should_apply_domain_confusion(self, domain_label):
        return self.training and torch.unique(domain_label).numel() > 1

    def _upsample_to_image(self, logits):
        return F.interpolate(
            logits,
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False,
        )

    def _weighted_bce_loss(self, logits, mask, domain_label, generative_weight=1.0):
        loss_map = self.BCE_loss(logits, mask)
        sample_weight = torch.where(
            domain_label.view(-1, 1, 1, 1) == 0,
            torch.ones_like(loss_map),
            torch.full_like(loss_map, generative_weight),
        )
        return (loss_map * sample_weight).mean()

    def _soft_dice_loss(self, logits, mask, sample_weight=None):
        prob = torch.sigmoid(logits)
        mask = mask.float()

        if sample_weight is None:
            sample_weight = torch.ones_like(mask)
        else:
            sample_weight = sample_weight.float()

        intersection = (prob * mask * sample_weight).sum(dim=(1, 2, 3))
        denominator = ((prob + mask) * sample_weight).sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
        return 1.0 - dice.mean()

    def _main_segmentation_loss(self, logits, mask, domain_label):
        bce = self._weighted_bce_loss(
            logits, mask, domain_label, generative_weight=self.current_generative_main_weight
        )
        sample_weight = torch.where(
            domain_label.view(-1, 1, 1, 1) == 0,
            torch.ones_like(mask),
            torch.full_like(mask, self.current_generative_main_weight),
        )
        dice = self._soft_dice_loss(logits, mask, sample_weight=sample_weight)
        return bce + self.dice_loss_weight * dice

    def _domain_specific_aux_loss(self, pred_logits, mask, domain_label, target_domain):
        selected = (domain_label == target_domain)
        if not selected.any():
            return pred_logits.new_zeros(())
        selected_logits = pred_logits[selected]
        selected_mask = mask[selected]
        bce = self.BCE_loss(selected_logits, selected_mask).mean()
        sample_weight = torch.ones_like(selected_mask)
        dice = self._soft_dice_loss(selected_logits, selected_mask, sample_weight=sample_weight)
        return bce + self.dice_loss_weight * dice

    def _dilate_binary_mask(self, mask, kernel_size):
        if kernel_size <= 1:
            return mask
        padding = kernel_size // 2
        return F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=padding)

    def _generative_aux_loss(self, pred_logits, mask, domain_label):
        selected = (domain_label == 1)
        if not selected.any():
            return pred_logits.new_zeros(())
        selected_logits = pred_logits[selected]
        selected_mask = mask[selected].float()
        dilated_mask = self._dilate_binary_mask(selected_mask, self.generative_aux_dilation_kernel)
        bce = self.BCE_loss(selected_logits, dilated_mask).mean()
        dice = self._soft_dice_loss(selected_logits, dilated_mask, sample_weight=torch.ones_like(dilated_mask))
        return bce + self.dice_loss_weight * dice

    def _forward_outputs(self, image, domain_label=None):
        batch_size = image.size(0)

        if domain_label is None:
            domain_label = torch.zeros(batch_size, device=image.device, dtype=torch.long)
        else:
            domain_label = domain_label.to(image.device).long()

        encoder_outputs = self.encoder_net(image)
        decoder_features = self._get_decoder_features(encoder_outputs)

        shallow_feat = decoder_features['third1']
        mid_feat = decoder_features['third']
        high_feat = decoder_features['last']
        sensitive_shallow_feat = (
            self.nonsemantic_shallow_adapter(shallow_feat)
            if self.use_nonsemantic_adapter
            else shallow_feat
        )
        sensitive_mid_feat = (
            self.nonsemantic_mid_adapter(mid_feat)
            if self.use_nonsemantic_adapter
            else mid_feat
        )
        invariant_high_feat = (
            self.semantic_context_adapter(high_feat)
            if self.use_semantic_adapter
            else high_feat
        )

        outputs = {
            'base_logits': None,
            'invariant_main_logits': None,
            'sensitive_main_logits': None,
            'logits': None,
            'pred_prob': None,
            'sensitive_feat': None,
            'invariant_feat': None,
            'freq_feat': None,
            'sensitive_scale': None,
        }

        base_logits = self.lmu(decoder_features)
        outputs['base_logits'] = base_logits

        sensitive_feat = self.sensitive_branch(sensitive_shallow_feat, sensitive_mid_feat)
        outputs['sensitive_feat'] = sensitive_feat
        raw_sensitive_main_logits = self.sensitive_main_head(sensitive_feat)
        sensitive_main_logits = self.sensitive_main_logit_bound * torch.tanh(
            raw_sensitive_main_logits / self.sensitive_main_logit_bound
        )
        outputs['sensitive_main_logits'] = sensitive_main_logits

        if self.stage_name != 'stage1':
            invariant_feat = self.invariant_branch(mid_feat, invariant_high_feat)

            freq_scale = self._get_frequency_injection_scale()
            outputs['frequency_injection_scale'] = freq_scale
            if freq_scale > 0.0:
                freq_feat = self.frequency_branch(image, target_size=invariant_feat.shape[2:])
                invariant_feat = self.invariant_fuse(
                    torch.cat([invariant_feat, freq_scale * freq_feat], dim=1)
                )
                outputs['freq_feat'] = freq_feat
            outputs['invariant_feat'] = invariant_feat
            raw_invariant_main_logits = self.invariant_main_head(invariant_feat)
            invariant_main_logits = self.invariant_main_logit_bound * torch.tanh(
                raw_invariant_main_logits / self.invariant_main_logit_bound
            )
            outputs['invariant_main_logits'] = invariant_main_logits
            fusion_scale = self._get_main_fusion_scale()
            outputs['main_fusion_scale'] = fusion_scale
            sensitive_scale = self._get_sensitive_main_fusion_scale(domain_label).type_as(base_logits)
            outputs['sensitive_scale'] = sensitive_scale
            fused_logits = base_logits + sensitive_scale * sensitive_main_logits + fusion_scale * invariant_main_logits
        else:
            sensitive_scale = self._get_sensitive_main_fusion_scale(domain_label).type_as(base_logits)
            outputs['sensitive_scale'] = sensitive_scale
            fused_logits = base_logits + sensitive_scale * sensitive_main_logits
            outputs['main_fusion_scale'] = 0.0
            outputs['frequency_injection_scale'] = 0.0

        logits = self._upsample_to_image(fused_logits)
        pred_prob = torch.sigmoid(logits)
        outputs['logits'] = logits
        outputs['pred_prob'] = pred_prob

        return outputs, domain_label

    def forward(self, image, mask=None, domain_label=None, return_aux=False, compute_loss=True):
        outputs, domain_label = self._forward_outputs(image, domain_label)
        logits = outputs['logits']
        pred_prob = outputs['pred_prob']

        if not compute_loss:
            if return_aux:
                return outputs
            return pred_prob

        predict_loss = self._main_segmentation_loss(logits, mask, domain_label)

        if self.stage_name == 'stage1':
            sensitive_feat = outputs['sensitive_feat']
            traditional_aux_logits = self.traditional_aux_head(sensitive_feat)
            traditional_aux_pred = self._upsample_to_image(traditional_aux_logits)
            traditional_aux_loss = self._domain_specific_aux_loss(
                traditional_aux_pred, mask, domain_label, target_domain=0
            )
            generative_aux_loss = logits.new_zeros(())
            confusion_loss = logits.new_zeros(())
            decouple_loss = logits.new_zeros(())
        else:
            sensitive_feat = outputs['sensitive_feat']
            invariant_feat = outputs['invariant_feat']

            traditional_aux_logits = self.traditional_aux_head(sensitive_feat)
            generative_aux_logits = self.generative_aux_head(invariant_feat)

            traditional_aux_pred = self._upsample_to_image(traditional_aux_logits)
            generative_aux_pred = self._upsample_to_image(generative_aux_logits)

            traditional_aux_loss = self._domain_specific_aux_loss(
                traditional_aux_pred, mask, domain_label, target_domain=0
            )
            generative_aux_loss = self._generative_aux_loss(
                generative_aux_pred, mask, domain_label
            )

            if self.stage_name == 'stage2' and self._should_apply_domain_confusion(domain_label):
                inv_global = invariant_feat.mean(dim=(2, 3))
                inv_global = self.grl_scalar(inv_global)
                confusion_logits = self.confusion_head_global(inv_global)
                confusion_loss = self.domain_loss(confusion_logits, domain_label)
            else:
                confusion_loss = logits.new_zeros(())

            decouple_loss = self.feature_decouple_loss(
                sensitive_feat,
                invariant_feat,
            )

        total_loss = (
            predict_loss
            + self.sensitive_loss_weight * traditional_aux_loss
            + self.invariant_loss_weight * generative_aux_loss
            + self.confusion_loss_weight * confusion_loss
            + self.decouple_loss_weight * decouple_loss
        )

        if return_aux:
            return total_loss, pred_prob, outputs
        return total_loss, pred_prob

