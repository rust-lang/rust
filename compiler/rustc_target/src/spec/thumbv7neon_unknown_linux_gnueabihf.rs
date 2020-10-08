use crate::spec::{Target, TargetOptions};

// This target is for glibc Linux on ARMv7 with thumb mode enabled
// (for consistency with Android and Debian-based distributions)
// and with NEON unconditionally enabled and, therefore, with 32 FPU
// registers enabled as well. See section A2.6.2 on page A2-56 in
// https://static.docs.arm.com/ddi0406/cd/DDI0406C_d_armv7ar_arm.pdf

pub fn target() -> Target {
    let base = super::linux_base::opts();
    Target {
        llvm_target: "armv7-unknown-linux-gnueabihf".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),

        options: TargetOptions {
            // Info about features at https://wiki.debian.org/ArmHardFloatPort
            features: "+v7,+thumb-mode,+thumb2,+vfp3,+neon".to_string(),
            cpu: "generic".to_string(),
            max_atomic_width: Some(64),
            unsupported_abis: super::arm_base::unsupported_abis(),
            ..base
        },
    }
}
