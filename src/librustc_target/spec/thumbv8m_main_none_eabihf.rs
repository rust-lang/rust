// Targets the Cortex-M33 processor (Armv8-M Mainline architecture profile),
// with the Floating Point extension.

use crate::spec::{LinkerFlavor, LldFlavor, Target, TargetOptions, TargetResult};

pub fn target() -> TargetResult {
    Ok(Target {
        llvm_target: "thumbv8m.main-none-eabihf".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        target_os: "none".to_string(),
        target_env: String::new(),
        target_vendor: String::new(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),

        options: TargetOptions {
            // If the Floating Point extension is implemented in the Cortex-M33
            // processor, the Cortex-M33 Technical Reference Manual states that
            // the FPU uses the FPv5 architecture, single-precision instructions
            // and 16 D registers.
            // These parameters map to the following LLVM features.
            features: "+fp-armv8,+fp-only-sp,+d16".to_string(),
            max_atomic_width: Some(32),
            .. super::thumb_base::opts()
        },
    })
}
