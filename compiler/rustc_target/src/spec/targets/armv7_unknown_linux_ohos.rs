use crate::spec::{FloatAbi, Target, TargetMetadata, TargetOptions, base};

// This target is for OpenHarmony on ARMv7 Linux with thumb-mode, but no NEON or
// hardfloat.

pub(crate) fn target() -> Target {
    // Most of these settings are copied from the armv7_unknown_linux_musleabi
    // target.
    Target {
        llvm_target: "armv7-unknown-linux-ohos".into(),
        metadata: TargetMetadata {
            description: Some("Armv7-A OpenHarmony".into()),
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
            features: "+v7,+thumb2,+soft-float,-neon".into(),
            max_atomic_width: Some(64),
            mcount: "\u{1}mcount".into(),
            ..base::linux_ohos::opts()
        },
    }
}
