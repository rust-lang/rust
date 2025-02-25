use crate::spec::{FloatAbi, Target, TargetMetadata, TargetOptions, base};

// This target is for uclibc Linux on ARMv7 without NEON,
// thumb-mode or hardfloat.

pub(crate) fn target() -> Target {
    let base = base::linux_uclibc::opts();
    Target {
        llvm_target: "armv7-unknown-linux-gnueabi".into(),
        metadata: TargetMetadata {
            description: Some("Armv7-A Linux with uClibc, softfloat".into()),
            tier: Some(3),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),

        options: TargetOptions {
            abi: "eabi".into(),
            llvm_floatabi: Some(FloatAbi::Soft),
            features: "+v7,+thumb2,+soft-float,-neon".into(),
            cpu: "generic".into(),
            max_atomic_width: Some(64),
            mcount: "_mcount".into(),
            ..base
        },
    }
}
