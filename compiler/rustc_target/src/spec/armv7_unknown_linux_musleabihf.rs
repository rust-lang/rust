use crate::spec::{Target, TargetOptions};

// This target is for musl Linux on ARMv7 without thumb-mode or NEON.

pub fn target() -> Target {
    let base = super::linux_musl_base::opts();
    Target {
        // It's important we use "gnueabihf" and not "musleabihf" here. LLVM
        // uses it to determine the calling convention and float ABI, and LLVM
        // doesn't support the "musleabihf" value.
        llvm_target: "armv7-unknown-linux-gnueabihf".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),

        // Most of these settings are copied from the armv7_unknown_linux_gnueabihf
        // target.
        options: TargetOptions {
            features: "+v7,+vfp3,-d32,+thumb2,-neon".to_string(),
            cpu: "generic".to_string(),
            max_atomic_width: Some(64),
            unsupported_abis: super::arm_base::unsupported_abis(),
            mcount: "\u{1}mcount".to_string(),
            ..base
        },
    }
}
