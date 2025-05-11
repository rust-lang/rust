use crate::spec::{SanitizerSet, StackProbeType, Target, base};

pub(crate) fn target() -> Target {
    let mut base = base::lynxos178::opts();
    base.cpu = "x86-64".into();
    base.plt_by_default = false;
    base.max_atomic_width = Some(64);
    base.stack_probes = StackProbeType::Inline;
    base.static_position_independent_executables = false;
    base.supported_sanitizers = SanitizerSet::ADDRESS
        | SanitizerSet::CFI
        | SanitizerSet::KCFI
        | SanitizerSet::DATAFLOW
        | SanitizerSet::LEAK
        | SanitizerSet::MEMORY
        | SanitizerSet::SAFESTACK
        | SanitizerSet::THREAD;
    base.supports_xray = true;

    Target {
        llvm_target: "x86_64-unknown-unknown-gnu".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("LynxOS-178".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 64,
        data_layout:
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        options: base,
    }
}
