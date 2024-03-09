use crate::spec::base::apple::{macos_llvm_target, opts, Arch};
use crate::spec::{Cc, FramePointer, LinkerFlavor, Lld, SanitizerSet};
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    let arch = Arch::X86_64h;
    let mut base = opts("macos", arch);
    base.max_atomic_width = Some(128);
    base.frame_pointer = FramePointer::Always;
    base.add_pre_link_args(LinkerFlavor::Darwin(Cc::Yes, Lld::No), &["-m64"]);
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
        llvm_target: macos_llvm_target(arch).into(),
        description: None,
        pointer_width: 64,
        data_layout:
            "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: arch.target_arch(),
        options: TargetOptions { mcount: "\u{1}mcount".into(), ..base },
    }
}
