// Targets the Cortex-M3 processor (ARMv7-M)

use crate::spec::{LinkerFlavor, LldFlavor, Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "thumbv7m-none-eabi".to_string(),
        target_endian: "little".to_string(),
        pointer_width: 32,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        target_os: "none".to_string(),
        target_env: String::new(),
        target_vendor: String::new(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),

        options: TargetOptions { max_atomic_width: Some(32), ..super::thumb_base::opts() },
    }
}
