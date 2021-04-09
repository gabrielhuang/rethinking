from .config import add_troi_config
from .mybox_head import MyFastRCNNTransformerHead
from .mypooler import MyROIPooler
from .troi_heads import TransformerROIHeads
from .myfpn import build_resnet_myfpn_backbone, build_resnet_myfpn_backbone_v2, \
    build_resnet_mybifpn_backbone, build_resnet_mybifpn_backbone_v2, build_resnet_myfpn_backbone_p4
from .myrpn import MyStandardRPNHead
from .dataset_mapper import DetrDatasetMapper
from .add_synthetic_dataset import synthetic_dataset_func
