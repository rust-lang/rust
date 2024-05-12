use crate::spec::base::apple::{macos_llvm_target, opts, pre_link_args, Arch, TargetAbi};
use crate::spec::{Cc, FramePointer, LinkerFlavor, Lld, MaybeLazy, SanitizerSet};
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    const ARCH: Arch = Arch::X86_64h;
    const OS: &'static str = "macos";
    const ABI: TargetAbi = TargetAbi::Normal;

    let mut base = opts(OS, ARCH, ABI, MaybeLazy::lazy(|| pre_link_args(OS, ARCH, ABI)));
    base.max_atomic_width = Some(128);
    base.frame_pointer = FramePointer::Always;
    base.pre_link_args = MaybeLazy::lazy(|| {
        TargetOptions::link_args(LinkerFlavor::Darwin(Cc::Yes, Lld::No), &["-m64"])
    });
    base.supported_sanitizers =
        SanitizerSet::ADDRESS | SanitizerSet::CFI | SanitizerSet::LEAK | SanitizerSet::THREAD;

    // x86_64h is core2-avx without a few of the features which would otherwise
    // be guaranteed, so we need to disable those. This imitates clang's logic:
    // - https://github.com/llvm/llvm-project/blob/bd1f7c417/clang/lib/Driver/ToolChains/Arch/X86.cpp#L77-L78
    // - https://github.com/llvm/llvm-project/blob/bd1f7c417/clang/lib/Driver/ToolChains/Arch/X86.cpp#L133-L141
    //
    // FIXME: Sadly, turning these off here disables them in such a way that they
    // aren't re-enabled by `-Ctarget-cpu=native` (on a machine that has them).
    // It would be nice if this were not the case, but fixing it seems tricky
    // (and given that the main use-case for this target is for use in universal
    // binaries, probably not that important).
    base.features = "-rdrnd,-aes,-pclmul,-rtm,-fsgsbase".into();
    // Double-check that the `cpu` is what we expect (if it's not the list above
    // may need updating).
    assert_eq!(
        base.cpu, "core-avx2",
        "you need to adjust the feature list in x86_64h-apple-darwin if you change this",
    );

    Target {
        // Clang automatically chooses a more specific target based on
        // MACOSX_DEPLOYMENT_TARGET. To enable cross-language LTO to work
        // correctly, we do too.
        llvm_target: MaybeLazy::lazy(|| macos_llvm_target(ARCH)),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 64,
        data_layout:
            "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: ARCH.target_arch(),
        options: TargetOptions { mcount: "\u{1}mcount".into(), ..base },
    }
}
