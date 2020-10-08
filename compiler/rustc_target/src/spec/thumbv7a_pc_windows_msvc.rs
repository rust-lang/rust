use crate::spec::{LinkerFlavor, LldFlavor, PanicStrategy, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::windows_msvc_base::opts();

    // Prevent error LNK2013: BRANCH24(T) fixup overflow
    // The LBR optimization tries to eliminate branch islands,
    // but if the displacement is larger than can fit
    // in the instruction, this error will occur. The linker
    // should be smart enough to insert branch islands only
    // where necessary, but this is not the observed behavior.
    // Disabling the LBR optimization works around the issue.
    let pre_link_args_msvc = "/OPT:NOLBR".to_string();
    base.pre_link_args.get_mut(&LinkerFlavor::Msvc).unwrap().push(pre_link_args_msvc.clone());
    base.pre_link_args
        .get_mut(&LinkerFlavor::Lld(LldFlavor::Link))
        .unwrap()
        .push(pre_link_args_msvc);

    // FIXME(jordanrh): use PanicStrategy::Unwind when SEH is
    // implemented for windows/arm in LLVM
    base.panic_strategy = PanicStrategy::Abort;

    Target {
        llvm_target: "thumbv7a-pc-windows-msvc".to_string(),
        pointer_width: 32,
        data_layout: "e-m:w-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),

        options: TargetOptions {
            features: "+vfp3,+neon".to_string(),
            cpu: "generic".to_string(),
            max_atomic_width: Some(64),
            unsupported_abis: super::arm_base::unsupported_abis(),
            ..base
        },
    }
}
