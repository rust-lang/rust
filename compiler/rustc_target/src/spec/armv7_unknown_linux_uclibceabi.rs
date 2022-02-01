use crate::spec::{Target, TargetOptions};

// This target is for uclibc Linux on ARMv7 without NEON,
// thumb-mode or hardfloat.

pub fn target() -> Target {
    let base = super::linux_uclibc_base::opts();
    Target {
        llvm_target: "armv7-unknown-linux-gnueabi".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),

        options: TargetOptions {
            features: "+v7,+thumb2,+soft-float,-neon".to_string(),
            cpu: "generic".to_string(),
            max_atomic_width: Some(64),
            mcount: "_mcount".to_string(),
            abi: "eabi".to_string(),
            ..base
        },
    }
}
