use super::{RelocModel, Target, TargetOptions};

pub fn target() -> Target {
    let base = super::solid_base::opts("asp3");
    Target {
        llvm_target: "armv7a-none-eabihf".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        options: TargetOptions {
            linker: Some("arm-kmc-eabi-gcc".to_owned()),
            features: "+v7,+vfp3,-d32,+thumb2,-neon".to_string(),
            relocation_model: RelocModel::Static,
            disable_redzone: true,
            max_atomic_width: Some(64),
            ..base
        },
    }
}
