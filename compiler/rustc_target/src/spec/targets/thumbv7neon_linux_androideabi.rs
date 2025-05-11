use crate::spec::{Cc, FloatAbi, LinkerFlavor, Lld, Target, TargetMetadata, TargetOptions, base};

// This target if is for the Android v7a ABI in thumb mode with
// NEON unconditionally enabled and, therefore, with 32 FPU registers
// enabled as well. See section A2.6.2 on page A2-56 in
// https://web.archive.org/web/20210307234416/https://static.docs.arm.com/ddi0406/cd/DDI0406C_d_armv7ar_arm.pdf

// See https://developer.android.com/ndk/guides/abis.html#v7a
// for target ABI requirements.

pub(crate) fn target() -> Target {
    let mut base = base::android::opts();
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-march=armv7-a"]);
    Target {
        llvm_target: "armv7-none-linux-android".into(),
        metadata: TargetMetadata {
            description: Some("Thumb2-mode ARMv7-A Android with NEON".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: TargetOptions {
            abi: "eabi".into(),
            llvm_floatabi: Some(FloatAbi::Soft),
            features: "+v7,+thumb-mode,+thumb2,+vfp3,+neon".into(),
            max_atomic_width: Some(64),
            ..base
        },
    }
}
