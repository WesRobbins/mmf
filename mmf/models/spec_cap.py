from mmf.common.registry import registry
from mmf.models.m4c_captioner import M4CCaptioner


@registry.register_model("spec_cap")
class SpecCap(M4CCaptioner):
    def __init__(self, config):
        super().__init__(config)
