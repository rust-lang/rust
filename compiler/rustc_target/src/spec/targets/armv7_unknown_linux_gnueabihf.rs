use crate::spec::{base, Target, TargetOptions};

// This target is for glibc Linux on ARMv7 without NEON or
// thumb-mode. See the thumbv7neon variant for enabling both.

pub fn target() -> Target {
    Target {
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
        options: TargetOptions {
            abi: "eabihf".into(),
            // Info about features at https://wiki.debian.org/ArmHardFloatPort
            features: "+v7,+vfp3,-d32,+thumb2,-neon".into(),
            max_atomic_width: Some(64),
            mcount: "\u{1}__gnu_mcount_nc".into(),
            ..base::linux_gnu::opts()
        },
    }
}
