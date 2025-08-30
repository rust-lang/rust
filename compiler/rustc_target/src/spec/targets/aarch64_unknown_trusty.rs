// Trusty OS target for AArch64.

use crate::spec::{
    LinkSelfContainedDefault, PanicStrategy, RelroLevel, Target, TargetMetadata, TargetOptions,
};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "aarch64-unknown-unknown-musl".into(),
        metadata: TargetMetadata {
            description: Some("ARM64 Trusty".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "aarch64".into(),
        options: TargetOptions {
            features: "+neon,+fp-armv8,+reserve-x18".into(),
            executables: true,
            max_atomic_width: Some(128),
            panic_strategy: PanicStrategy::Abort,
            os: "trusty".into(),
            position_independent_executables: true,
            static_position_independent_executables: true,
            crt_static_default: true,
            crt_static_respected: true,
            dynamic_linking: false,
            link_self_contained: LinkSelfContainedDefault::InferredForMusl,
            relro_level: RelroLevel::Full,
            mcount: "\u{1}_mcount".into(),
            ..Default::default()
        },
    }
}
