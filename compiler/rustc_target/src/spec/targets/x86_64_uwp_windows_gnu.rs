use crate::spec::{Cc, LinkerFlavor, Lld, Target, base};

pub(crate) fn target() -> Target {
    let mut base = base::windows_uwp_gnu::opts();
    base.cpu = "x86-64".into();
    base.features = "+cx16,+sse3,+sahf".into();
    base.plt_by_default = false;
    // Use high-entropy 64 bit address space for ASLR
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::No, Lld::No), &[
        "-m",
        "i386pep",
        "--high-entropy-va",
    ]);
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-m64", "-Wl,--high-entropy-va"]);
    base.max_atomic_width = Some(128);

    Target {
        llvm_target: "x86_64-pc-windows-gnu".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout:
            "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        options: base,
    }
}
