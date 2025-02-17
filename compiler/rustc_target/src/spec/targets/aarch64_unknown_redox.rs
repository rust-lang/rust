use crate::spec::{StackProbeType, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::redox::opts();
    base.max_atomic_width = Some(128);
    base.stack_probes = StackProbeType::Inline;
    base.features = "+v8a".into();

    Target {
        llvm_target: "aarch64-unknown-redox".into(),
        metadata: TargetMetadata {
            description: Some("ARM64 RedoxOS".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: None, // ?
        },
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "aarch64".into(),
        options: base,
    }
}
