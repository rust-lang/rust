use crate::spec::{Target, TargetMetadata};

pub(crate) fn target() -> Target {
    let mut base = super::riscv64gc_unknown_linux_gnu::target();

    base.metadata = TargetMetadata {
        description: Some("RISC-V Linux (kernel 4.20, glibc 2.29) for yocto".into()),
        tier: Some(3),
        host_tools: Some(false),
        std: Some(true),
    };

    base.llvm_target = "riscv64-oe-linux-gnu".into();
    base.options.linker = Some("riscv64-oe-linux-gcc".into());

    base
}
