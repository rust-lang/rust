use crate::spec::{Arch, Cc, LinkerFlavor, Lld, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    let mut base = TargetOptions {
        vendor: "win7".into(),
        cpu: "x86-64".into(),
        plt_by_default: false,
        max_atomic_width: Some(64),
        linker: Some("x86_64-w64-mingw32-gcc".into()),
        ..base::windows_gnu::opts()
    };
    // Use high-entropy 64 bit address space for ASLR
    base.add_pre_link_args(
        LinkerFlavor::Gnu(Cc::No, Lld::No),
        &["-m", "i386pep", "--high-entropy-va"],
    );
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-m64", "-Wl,--high-entropy-va"]);

    Target {
        llvm_target: "x86_64-pc-windows-gnu".into(),
        metadata: TargetMetadata {
            description: Some("64-bit MinGW (Windows 7+)".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout:
            "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: Arch::X86_64,
        options: base,
    }
}
