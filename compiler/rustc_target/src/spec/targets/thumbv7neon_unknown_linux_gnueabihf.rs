use crate::spec::{FloatAbi, Target, TargetMetadata, TargetOptions, base};

// This target is for glibc Linux on ARMv7 with thumb mode enabled
// (for consistency with Android and Debian-based distributions)
// and with NEON unconditionally enabled and, therefore, with 32 FPU
// registers enabled as well. See section A2.6.2 on page A2-56 in
// https://web.archive.org/web/20210307234416/https://static.docs.arm.com/ddi0406/cd/DDI0406C_d_armv7ar_arm.pdf

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "armv7-unknown-linux-gnueabihf".into(),
        metadata: TargetMetadata {
            description: Some(
                "Thumb2-mode ARMv7-A Linux with NEON (kernel 4.4, glibc 2.23)".into(),
            ),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: TargetOptions {
            abi: "eabihf".into(),
            llvm_floatabi: Some(FloatAbi::Hard),
            // Info about features at https://wiki.debian.org/ArmHardFloatPort
            features: "+v7,+thumb-mode,+thumb2,+vfp3,+neon".into(),
            max_atomic_width: Some(64),
            ..base::linux_gnu::opts()
        },
    }
}
