use crate::spec::{Target, TargetOptions};

// This target is for uclibc Linux on ARMv7 without NEON or
// thumb-mode. See the thumbv7neon variant for enabling both.

pub fn target() -> Target {
    let base = super::linux_uclibc_base::opts();
    Target {
        llvm_target: "armv7-unknown-linux-gnueabihf".into(),
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
            ..base
        },
    }
}
