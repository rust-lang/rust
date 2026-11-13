use crate::spec::{Target, TargetMetadata, TargetOptions};

pub(crate) fn target() -> Target {
    let mut base = super::aarch64_unknown_linux_gnu::target();

    base.metadata = TargetMetadata {
        description: Some("ARM64 Vega OS (glibc 2.35)".into()),
        tier: Some(3),
        host_tools: Some(false),
        std: Some(true),
    };

    base.options = TargetOptions { vendor: "vega".into(), ..base.options };

    base
}
