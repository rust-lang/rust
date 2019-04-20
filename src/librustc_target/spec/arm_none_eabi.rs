// Generic arm target for bare-metal code
//
// Can be used in conjunction with the `target-feature` and
// `target-cpu` compiler flags to opt-in more hardware-specific
// features.
//
// For example, `-C target-cpu=cortex-a7`.

use super::{LinkerFlavor, Target, TargetOptions, PanicStrategy};

pub fn target() -> Result<Target, String> {
    Ok(Target {
        llvm_target: "arm-none-eabi".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        target_os: "none".to_string(),
        target_env: String::new(),
        target_vendor: String::new(),
        linker_flavor: LinkerFlavor::Gcc,

        options: TargetOptions {
            features: "+strict-align,+v6".to_string(),
            executables: true,
            relocation_model: "static".to_string(),
            max_atomic_width: Some(64),
            panic_strategy: PanicStrategy::Abort,
            abi_blacklist: super::arm_base::abi_blacklist(),
            .. Default::default()
        },
    })
}
