use crate::spec::{Target, TargetMetadata};

pub(crate) fn target() -> Target {
    let mut base = super::armv7_unknown_linux_gnueabihf::target();

    base.metadata = TargetMetadata {
        description: Some("Armv7-A Linux, hardfloat (kernel 3.2, glibc 2.17) for yocto".into()),
        tier: Some(3),
        host_tools: Some(false),
        std: Some(true),
    };

    base.llvm_target = "armv7-oe-linux-gnueabihf".into();
    base.options.linker = Some("arm-oe-linux-gnueabi-gcc".into());

    base
}
