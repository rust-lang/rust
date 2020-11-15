use crate::spec::{PanicStrategy, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::windows_uwp_msvc_base::opts();
    base.max_atomic_width = Some(64);
    base.has_elf_tls = true;

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
            unsupported_abis: super::arm_base::unsupported_abis(),
            ..base
        },
    }
}
