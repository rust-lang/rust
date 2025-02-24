use crate::spec::{Cc, FramePointer, LinkerFlavor, Lld, RustcAbi, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::windows_gnullvm::opts();
    base.rustc_abi = Some(RustcAbi::X86Sse2);
    base.cpu = "pentium4".into();
    base.max_atomic_width = Some(64);
    base.frame_pointer = FramePointer::Always; // Required for backtraces
    base.linker = Some("i686-w64-mingw32-clang".into());

    // Mark all dynamic libraries and executables as compatible with the larger 4GiB address
    // space available to x86 Windows binaries on x86_64.
    base.add_pre_link_args(
        LinkerFlavor::Gnu(Cc::No, Lld::No),
        &["-m", "i386pe", "--large-address-aware"],
    );

    Target {
        llvm_target: "i686-pc-windows-gnu".into(),
        metadata: TargetMetadata {
            description: Some("32-bit x86 MinGW (Windows 10+), LLVM ABI".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i64:64-i128:128-f80:32-n8:16:32-a:0:32-S32"
            .into(),
        arch: "x86".into(),
        options: base,
    }
}
