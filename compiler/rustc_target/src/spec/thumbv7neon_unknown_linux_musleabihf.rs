use crate::spec::{Target, TargetOptions};

// This target is for musl Linux on ARMv7 with thumb mode enabled
// (for consistency with Android and Debian-based distributions)
// and with NEON unconditionally enabled and, therefore, with 32 FPU
// registers enabled as well. See section A2.6.2 on page A2-56 in
// https://web.archive.org/web/20210307234416/https://static.docs.arm.com/ddi0406/cd/DDI0406C_d_armv7ar_arm.pdf

pub fn target() -> Target {
    Target {
        // It's important we use "gnueabihf" and not "musleabihf" here. LLVM
        // uses it to determine the calling convention and float ABI, and LLVM
        // doesn't support the "musleabihf" value.
        llvm_target: "armv7-unknown-linux-gnueabihf".into(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),

        // Most of these settings are copied from the thumbv7neon_unknown_linux_gnueabihf
        // target.
        options: TargetOptions {
            abi: "eabihf".into(),
            features: "+v7,+thumb-mode,+thumb2,+vfp3,+neon".into(),
            max_atomic_width: Some(64),
            mcount: "\u{1}mcount".into(),
            ..super::linux_musl_base::opts()
        },
    }
}
