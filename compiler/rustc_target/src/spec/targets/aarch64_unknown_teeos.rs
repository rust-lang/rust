use crate::spec::{base, StackProbeType, Target};

pub fn target() -> Target {
    let mut base = base::teeos::opts();
    base.features = "+strict-align,+neon,+fp-armv8".into();
    base.max_atomic_width = Some(128);
    base.stack_probes = StackProbeType::Inline;

    Target {
        llvm_target: "aarch64-unknown-none".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 64,
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".into(),
        arch: "aarch64".into(),
        options: base,
    }
}
