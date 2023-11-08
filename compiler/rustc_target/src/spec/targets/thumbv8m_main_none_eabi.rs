// Targets the Cortex-M33 processor (Armv8-M Mainline architecture profile),
// without the Floating Point extension.

use crate::spec::{base, Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "thumbv8m.main-none-eabi".into(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),

        options: TargetOptions {
            abi: "eabi".into(),
            max_atomic_width: Some(32),
            ..base::thumb::opts()
        },
    }
}
