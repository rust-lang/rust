use crate::spec::{FloatAbi, Target, TargetMetadata, TargetOptions, base};

// This target is for glibc Linux on ARMv7 without thumb-mode, NEON or
// hardfloat.

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "armv7-unknown-linux-gnueabi".into(),
        metadata: TargetMetadata {
            description: Some("Armv7-A Linux (kernel 4.15, glibc 2.27)".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: TargetOptions {
            abi: "eabi".into(),
            llvm_floatabi: Some(FloatAbi::Soft),
            features: "+v7,+thumb2,+soft-float,-neon".into(),
            max_atomic_width: Some(64),
            mcount: "\u{1}__gnu_mcount_nc".into(),
            llvm_mcount_intrinsic: Some("llvm.arm.gnu.eabi.mcount".into()),
            ..base::linux_gnu::opts()
        },
    }
}
