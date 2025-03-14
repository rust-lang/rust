// Targets the Cortex-M33 processor (Armv8-M Mainline architecture profile),
// without the Floating Point extension.

use crate::spec::{FloatAbi, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "thumbv8m.main-none-eabi".into(),
        metadata: TargetMetadata {
            description: Some("Bare ARMv8-M Mainline".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),

        options: TargetOptions {
            abi: "eabi".into(),
            llvm_floatabi: Some(FloatAbi::Soft),
            max_atomic_width: Some(32),
            ..base::thumb::opts()
        },
    }
}
