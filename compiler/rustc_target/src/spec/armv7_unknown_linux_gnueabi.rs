use crate::spec::{Target, TargetOptions};

// This target is for glibc Linux on ARMv7 without thumb-mode, NEON or
// hardfloat.

pub fn target() -> Target {
    let base = super::linux_gnu_base::opts();
    Target {
        llvm_target: "armv7-unknown-linux-gnueabi".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),

        options: TargetOptions {
            features: "+v7,+thumb2,+soft-float,-neon".to_string(),
            cpu: "generic".to_string(),
            max_atomic_width: Some(64),
            unsupported_abis: super::arm_base::unsupported_abis(),
            mcount: "\u{1}__gnu_mcount_nc".to_string(),
            ..base
        },
    }
}
