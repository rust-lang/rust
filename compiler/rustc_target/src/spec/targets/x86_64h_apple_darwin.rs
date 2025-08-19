use crate::spec::base::apple::{Arch, TargetEnv, base};
use crate::spec::{SanitizerSet, Target, TargetMetadata, TargetOptions};

pub(crate) fn target() -> Target {
    let (mut opts, llvm_target, arch) = base("macos", Arch::X86_64h, TargetEnv::Normal);
    opts.max_atomic_width = Some(128);
    opts.supported_sanitizers =
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
    opts.features = "-rdrnd,-aes,-pclmul,-rtm,-fsgsbase".into();
    // Double-check that the `cpu` is what we expect (if it's not the list above
    // may need updating).
    assert_eq!(
        opts.cpu, "core-avx2",
        "you need to adjust the feature list in x86_64h-apple-darwin if you change this",
    );

    Target {
        llvm_target,
        metadata: TargetMetadata {
            description: Some("x86_64 Apple macOS with Intel Haswell+".into()),
            tier: Some(3),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout:
            "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch,
        options: TargetOptions { mcount: "\u{1}mcount".into(), ..opts },
    }
}
