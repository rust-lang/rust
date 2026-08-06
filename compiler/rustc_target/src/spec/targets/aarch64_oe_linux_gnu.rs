use crate::spec::{Target, TargetMetadata};

pub(crate) fn target() -> Target {
    let mut base = super::aarch64_unknown_linux_gnu::target();

    base.metadata = TargetMetadata {
        description: Some("64-bit Linux (kernel 3.2+, glibc 2.17+) for yocto".into()),
        tier: Some(3),
        host_tools: Some(false),
        std: Some(true),
    };

    base.llvm_target = "aarch64-oe-linux-gnu".into();
    base.options.linker = Some("aarch64-oe-linux-gcc".into());

    base
}
