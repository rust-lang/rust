use crate::spec::{base, LinkerFlavor, Lld, MaybeLazy, PanicStrategy, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = base::windows_msvc::opts();
    // Prevent error LNK2013: BRANCH24(T) fixup overflow
    // The LBR optimization tries to eliminate branch islands,
    // but if the displacement is larger than can fit
    // in the instruction, this error will occur. The linker
    // should be smart enough to insert branch islands only
    // where necessary, but this is not the observed behavior.
    // Disabling the LBR optimization works around the issue.
    base.pre_link_args =
        MaybeLazy::lazy(|| TargetOptions::link_args(LinkerFlavor::Msvc(Lld::No), &["/OPT:NOLBR"]));

    Target {
        llvm_target: "thumbv7a-pc-windows-msvc".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 32,
        data_layout: "e-m:w-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: TargetOptions {
            features: "+vfp3,+neon".into(),
            max_atomic_width: Some(64),
            // FIXME(jordanrh): use PanicStrategy::Unwind when SEH is
            // implemented for windows/arm in LLVM
            panic_strategy: PanicStrategy::Abort,
            ..base
        },
    }
}
