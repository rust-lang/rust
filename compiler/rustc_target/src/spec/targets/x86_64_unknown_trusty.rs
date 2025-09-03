// Trusty OS target for X86_64.

use crate::spec::{
    LinkSelfContainedDefault, PanicStrategy, RelroLevel, StackProbeType, Target, TargetMetadata,
    TargetOptions,
};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "x86_64-unknown-unknown-musl".into(),
        metadata: TargetMetadata {
            description: Some("x86_64 Trusty".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout:
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        options: TargetOptions {
            executables: true,
            max_atomic_width: Some(64),
            panic_strategy: PanicStrategy::Abort,
            os: "trusty".into(),
            link_self_contained: LinkSelfContainedDefault::InferredForMusl,
            position_independent_executables: true,
            static_position_independent_executables: true,
            crt_static_default: true,
            crt_static_respected: true,
            dynamic_linking: false,
            plt_by_default: false,
            relro_level: RelroLevel::Full,
            stack_probes: StackProbeType::Inline,
            mcount: "\u{1}_mcount".into(),
            ..Default::default()
        },
    }
}
