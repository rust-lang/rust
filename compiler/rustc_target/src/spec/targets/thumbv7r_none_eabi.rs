// Targets the Little-endian Cortex-R4/R5 processor (ARMv7-R)

use crate::spec::{
    Arch, CfgAbi, FloatAbi, StdCrates, StdSupport, Target, TargetMetadata, TargetOptions, base,
};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "thumbv7r-none-eabi".into(),
        metadata: TargetMetadata {
            description: Some("Thumb-mode Bare Armv7-R".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(false),
            standard_library_support: Some(StdSupport::from_crates(StdCrates::Alloc)),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: Arch::Arm,
        options: TargetOptions {
            cfg_abi: CfgAbi::Eabi,
            llvm_floatabi: Some(FloatAbi::Soft),
            max_atomic_width: Some(64),
            has_thumb_interworking: true,
            ..base::arm_none::opts()
        },
    }
}
