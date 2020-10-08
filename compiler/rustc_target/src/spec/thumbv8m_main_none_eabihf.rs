// Targets the Cortex-M33 processor (Armv8-M Mainline architecture profile),
// with the Floating Point extension.

use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "thumbv8m.main-none-eabihf".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),

        options: TargetOptions {
            // If the Floating Point extension is implemented in the Cortex-M33
            // processor, the Cortex-M33 Technical Reference Manual states that
            // the FPU uses the FPv5 architecture, single-precision instructions
            // and 16 D registers.
            // These parameters map to the following LLVM features.
            features: "+fp-armv8,-fp64,-d32".to_string(),
            max_atomic_width: Some(32),
            ..super::thumb_base::opts()
        },
    }
}
