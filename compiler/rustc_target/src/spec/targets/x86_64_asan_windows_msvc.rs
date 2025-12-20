use crate::spec::{SanitizerSet, Target, TargetMetadata};

pub(crate) fn target() -> Target {
    let mut base = super::x86_64_pc_windows_msvc::target();
    base.metadata = TargetMetadata {
        description: Some("64-bit Windows with ASAN enabled by default".into()),
        tier: Some(3),
        host_tools: Some(false),
        std: Some(true),
    };
    base.options.default_sanitizers = SanitizerSet::ADDRESS;

    assert!(base.options.default_sanitizers.contains(SanitizerSet::ADDRESS));

    base
}
