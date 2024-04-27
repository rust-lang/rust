use crate::spec::{base, Target, TargetOptions};

// This target is for musl Linux on ARMv7 without thumb-mode or NEON.

pub fn target() -> Target {
    Target {
        // It's important we use "gnueabihf" and not "musleabihf" here. LLVM
        // uses it to determine the calling convention and float ABI, and LLVM
        // doesn't support the "musleabihf" value.
        llvm_target: "armv7-unknown-linux-gnueabihf".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),

        // Most of these settings are copied from the armv7_unknown_linux_gnueabihf
        // target.
        options: TargetOptions {
            abi: "eabihf".into(),
            features: "+v7,+vfp3,-d32,+thumb2,-neon".into(),
            max_atomic_width: Some(64),
            mcount: "\u{1}mcount".into(),
            ..base::linux_musl::opts()
        },
    }
}
