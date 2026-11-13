use crate::spec::{SanitizerSet, Target, TargetMetadata};

pub(crate) fn target() -> Target {
    let mut base = super::x86_64_unknown_linux_gnu::target();
    base.metadata = TargetMetadata {
        description: Some(
            "64-bit Linux (kernel 3.2+, glibc 2.17+) with ASAN enabled by default".into(),
        ),
        tier: Some(2),
        host_tools: Some(false),
        std: Some(true),
    };
    base.supported_sanitizers = SanitizerSet::ADDRESS;
    base.default_sanitizers = SanitizerSet::ADDRESS;
    base
}
