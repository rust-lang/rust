use crate::spec::{StackProbeType, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "aarch64-unknown-hermit".into(),
        metadata: TargetMetadata {
            description: Some("ARM64 Hermit".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        arch: "aarch64".into(),
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        options: TargetOptions {
            features: "+v8a,+strict-align,+neon,+fp-armv8".into(),
            max_atomic_width: Some(128),
            stack_probes: StackProbeType::Inline,
            ..base::hermit::opts()
        },
    }
}
