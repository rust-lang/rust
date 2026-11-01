use crate::spec::{Target, TargetMetadata};

pub(crate) fn target() -> Target {
    let mut base = super::i686_unknown_linux_gnu::target();

    base.metadata = TargetMetadata {
        description: Some("32-bit Linux (kernel 3.2, glibc 2.17+) for yocto".into()),
        tier: Some(3),
        host_tools: Some(false),
        std: Some(true),
    };

    base.llvm_target = "i686-oe-linux-gnu".into();

    base.options.linker = Some("i686-oe-linux-gcc".into());

    base.options.cpu = "core2".into();
    base.options.features = "+sse3".into();

    base
}
