use crate::spec::{Target, TargetMetadata, TargetOptions};

pub(crate) fn target() -> Target {
    let mut base = super::x86_64_unknown_linux_gnu::target();

    base.metadata = TargetMetadata {
        description: Some("64-bit Vega OS (glibc 2.35)".into()),
        tier: Some(3),
        host_tools: Some(false),
        std: Some(true),
    };

    base.options = TargetOptions { vendor: "vega".into(), ..base.options };

    base
}
