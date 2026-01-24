// Targets the Little-endian Cortex-A8 (and similar) processors (ARMv7-A)

use crate::spec::{Abi, Arch, FloatAbi, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "armv7a-none-eabihf".into(),
        metadata: TargetMetadata {
            description: Some("Bare Armv7-A, hardfloat".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: Arch::Arm,
        options: TargetOptions {
            abi: Abi::EabiHf,
            llvm_floatabi: Some(FloatAbi::Hard),
            features: "+vfp3d16,-neon,+strict-align".into(),
            max_atomic_width: Some(64),
            has_thumb_interworking: true,
            ..base::arm_none::opts()
        },
    }
}
