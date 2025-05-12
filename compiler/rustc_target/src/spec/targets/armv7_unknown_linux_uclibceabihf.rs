use crate::spec::{FloatAbi, Target, TargetMetadata, TargetOptions, base};

// This target is for uclibc Linux on ARMv7 without NEON or
// thumb-mode. See the thumbv7neon variant for enabling both.

pub(crate) fn target() -> Target {
    let base = base::linux_uclibc::opts();
    Target {
        llvm_target: "armv7-unknown-linux-gnueabihf".into(),
        metadata: TargetMetadata {
            description: Some("Armv7-A Linux with uClibc, hardfloat".into()),
            tier: Some(3),
            host_tools: None, // ?
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),

        options: TargetOptions {
            // Info about features at https://wiki.debian.org/ArmHardFloatPort
            features: "+v7,+vfp3,-d32,+thumb2,-neon".into(),
            cpu: "generic".into(),
            max_atomic_width: Some(64),
            mcount: "_mcount".into(),
            abi: "eabihf".into(),
            llvm_floatabi: Some(FloatAbi::Hard),
            ..base
        },
    }
}
