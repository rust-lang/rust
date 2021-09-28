use super::{RelocModel, Target, TargetOptions};

pub fn target() -> Target {
    let base = super::solid_base::opts("asp3");
    Target {
        llvm_target: "aarch64-unknown-none".to_string(),
        pointer_width: 64,
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".to_string(),
        arch: "aarch64".to_string(),
        options: TargetOptions {
            linker: Some("aarch64-kmc-elf-gcc".to_owned()),
            features: "+neon,+fp-armv8".to_string(),
            relocation_model: RelocModel::Static,
            disable_redzone: true,
            max_atomic_width: Some(128),
            ..base
        },
    }
}
