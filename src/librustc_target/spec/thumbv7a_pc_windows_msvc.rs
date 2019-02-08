use crate::spec::{LinkerFlavor, Target, TargetOptions, TargetResult, PanicStrategy};

pub fn target() -> TargetResult {
    let mut base = super::windows_msvc_base::opts();

    // Prevent error LNK2013: BRANCH24(T) fixup overflow
    // The LBR optimization tries to eliminate branch islands,
    // but if the displacement is larger than can fit
    // in the instruction, this error will occur. The linker
    // should be smart enough to insert branch islands only
    // where necessary, but this is not the observed behavior.
    // Disabling the LBR optimization works around the issue.
    base.pre_link_args.get_mut(&LinkerFlavor::Msvc).unwrap().push(
        "/OPT:NOLBR".to_string());

    // FIXME(jordanrh): use PanicStrategy::Unwind when SEH is
    // implemented for windows/arm in LLVM
    base.panic_strategy = PanicStrategy::Abort;

    Ok(Target {
        llvm_target: "thumbv7a-pc-windows-msvc".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:w-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        target_os: "windows".to_string(),
        target_env: "msvc".to_string(),
        target_vendor: "pc".to_string(),
        linker_flavor: LinkerFlavor::Msvc,

        options: TargetOptions {
            features: "+vfp3,+neon".to_string(),
            cpu: "generic".to_string(),
            max_atomic_width: Some(64),
            abi_blacklist: super::arm_base::abi_blacklist(),
            .. base
        }
    })
}
