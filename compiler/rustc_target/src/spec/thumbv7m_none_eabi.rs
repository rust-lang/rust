// Targets the Cortex-M3 processor (ARMv7-M)

use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "thumbv7m-none-eabi".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),

        options: TargetOptions { max_atomic_width: Some(32), ..super::thumb_base::opts() },
    }
}
