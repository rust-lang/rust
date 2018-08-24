// Targets the Cortex-R4F/R5F processor (ARMv7-R)

use std::default::Default;
use spec::{LinkerFlavor, PanicStrategy, Target, TargetOptions, TargetResult};

pub fn target() -> TargetResult {
    Ok(Target {
        llvm_target: "armebv7r-none-eabihf".to_string(),
        target_endian: "big".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "E-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        target_os: "none".to_string(),
        target_env: "".to_string(),
        target_vendor: "".to_string(),
        linker_flavor: LinkerFlavor::Gcc,

        options: TargetOptions {
            executables: true,
            relocation_model: "static".to_string(),
            panic_strategy: PanicStrategy::Abort,
            features: "+v7,+vfp3,+d16,+fp-only-sp".to_string(),
            max_atomic_width: Some(32),
            abi_blacklist: super::arm_base::abi_blacklist(),
            emit_debug_gdb_scripts: false,
            .. Default::default()
        },
    })
}
