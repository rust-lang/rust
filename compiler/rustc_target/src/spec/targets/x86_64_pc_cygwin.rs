use crate::spec::{Cc, LinkerFlavor, Lld, Target, base};

pub(crate) fn target() -> Target {
    let mut base = base::cygwin::opts();
    base.cpu = "x86-64".into();
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::No, Lld::No), &["-m", "i386pep"]);
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-m64"]);
    base.max_atomic_width = Some(64);
    base.linker = Some("x86_64-pc-cygwin-gcc".into());
    Target {
        llvm_target: "x86_64-pc-cygwin".into(),
        pointer_width: 64,
        data_layout:
            "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        options: base,
        metadata: crate::spec::TargetMetadata {
            description: Some("64-bit x86 Cygwin".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
    }
}
