use crate::spec::{FloatAbi, RelocModel, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    let base = base::solid::opts("asp3");
    Target {
        llvm_target: "armv7a-none-eabihf".into(),
        metadata: TargetMetadata {
            description: Some("Arm SOLID with TOPPERS/ASP3, hardfloat".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: TargetOptions {
            abi: "eabihf".into(),
            llvm_floatabi: Some(FloatAbi::Hard),
            linker: Some("arm-kmc-eabi-gcc".into()),
            features: "+v7,+vfp3,-d32,+thumb2,-neon".into(),
            relocation_model: RelocModel::Static,
            disable_redzone: true,
            max_atomic_width: Some(64),
            ..base
        },
    }
}
