use crate::spec::{FloatAbi, Target, TargetMetadata, TargetOptions, base};

// This target is for glibc Linux on ARMv7 without NEON or
// thumb-mode. See the thumbv7neon variant for enabling both.

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "armv7-unknown-linux-gnueabihf".into(),
        metadata: TargetMetadata {
            description: Some("Armv7-A Linux, hardfloat (kernel 3.2, glibc 2.17)".into()),
            tier: Some(2),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: TargetOptions {
            abi: "eabihf".into(),
            llvm_floatabi: Some(FloatAbi::Hard),
            // Info about features at https://wiki.debian.org/ArmHardFloatPort
            features: "+v7,+vfp3,-d32,+thumb2,-neon".into(),
            max_atomic_width: Some(64),
            mcount: "\u{1}__gnu_mcount_nc".into(),
            llvm_mcount_intrinsic: Some("llvm.arm.gnu.eabi.mcount".into()),
            ..base::linux_gnu::opts()
        },
    }
}
