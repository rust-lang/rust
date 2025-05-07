use crate::spec::{FloatAbi, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "arm-unknown-linux-musleabi".into(),
        metadata: TargetMetadata {
            description: Some("Armv6 Linux with musl 1.2.3".into()),
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
            // Most of these settings are copied from the arm_unknown_linux_gnueabi
            // target.
            features: "+strict-align,+v6".into(),
            max_atomic_width: Some(64),
            mcount: "\u{1}mcount".into(),
            // FIXME(compiler-team#422): musl targets should be dynamically linked by default.
            crt_static_default: true,
            ..base::linux_musl::opts()
        },
    }
}
