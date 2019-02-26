// Generic AArch64 target for bare-metal code
//
// Can be used in conjunction with the `target-feature` and
// `target-cpu` compiler flags to opt-in more hardware-specific
// features.
//
// For example, `-C target-cpu=cortex-a53`.

use super::{LldFlavor, LinkerFlavor, Target, TargetOptions, PanicStrategy};

pub fn target() -> Result<Target, String> {
    let opts = TargetOptions {
        linker: Some("rust-lld".to_owned()),
        features: "+strict-align".to_string(),
        executables: true,
        relocation_model: "static".to_string(),
        disable_redzone: true,
        linker_is_gnu: true,
        max_atomic_width: Some(128),
        panic_strategy: PanicStrategy::Abort,
        abi_blacklist: super::arm_base::abi_blacklist(),
        .. Default::default()
    };
    Ok(Target {
        llvm_target: "aarch64-unknown-none".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "64".to_string(),
        target_c_int_width: "32".to_string(),
        target_os: "none".to_string(),
        target_env: String::new(),
        target_vendor: String::new(),
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".to_string(),
        arch: "aarch64".to_string(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
        options: opts,
    })
}
