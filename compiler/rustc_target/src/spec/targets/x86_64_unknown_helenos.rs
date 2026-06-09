use crate::spec::{Arch, Cc, LinkerFlavor, Lld, Target, base};

pub(crate) fn target() -> Target {
    let mut base = base::helenos::opts();
    base.cpu = "x86-64".into();
    base.plt_by_default = false;
    base.max_atomic_width = Some(64);
    base.linker = Some("amd64-helenos-gcc".into());
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-m64"]);

    Target {
        llvm_target: "x86_64-unknown-helenos".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("64-bit HelenOS".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout:
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: Arch::X86_64,
        options: base,
    }
}
