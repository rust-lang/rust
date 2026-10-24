use crate::spec::{Target, TargetMetadata};

pub(crate) fn target() -> Target {
    let mut base = super::powerpc64le_unknown_linux_gnu::target();

    base.metadata = TargetMetadata {
        description: Some(
            "PowerPC64 (little endian) Linux (kernel 3.10, glibc 2.17) for yocto".into(),
        ),
        tier: Some(3),
        host_tools: Some(false),
        std: Some(true),
    };

    base.llvm_target = "powerpc64le-oe-linux-gnu".into();
    base.options.linker = Some("powerpc64le-oe-linux-gcc".into());

    base
}
