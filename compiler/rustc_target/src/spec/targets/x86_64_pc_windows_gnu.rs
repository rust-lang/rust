use crate::spec::{add_link_args, base, Cc, LinkerFlavor, Lld, MaybeLazy, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = base::windows_gnu::opts();
    base.cpu = "x86-64".into();
    base.features = "+cx16,+sse3,+sahf".into();
    base.plt_by_default = false;
    // Use high-entropy 64 bit address space for ASLR
    base.pre_link_args = MaybeLazy::lazy(|| {
        let mut pre_link_args = TargetOptions::link_args_base(
            LinkerFlavor::Gnu(Cc::No, Lld::No),
            &["-m", "i386pep", "--high-entropy-va"],
        );
        add_link_args(
            &mut pre_link_args,
            LinkerFlavor::Gnu(Cc::Yes, Lld::No),
            &["-m64", "-Wl,--high-entropy-va"],
        );
        pre_link_args
    });
    base.max_atomic_width = Some(128);
    base.linker = Some("x86_64-w64-mingw32-gcc".into());

    Target {
        llvm_target: "x86_64-pc-windows-gnu".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 64,
        data_layout:
            "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        options: base,
    }
}
