// Targets the Cortex-M0, Cortex-M0+ and Cortex-M1 processors (ARMv6-M architecture)

use crate::spec::{FloatAbi, Target, TargetMetadata, TargetOptions, base, cvs};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "thumbv6m-none-eabi".into(),
        metadata: TargetMetadata {
            description: None,
            tier: Some(3),
            host_tools: None,
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),

        options: TargetOptions {
            families: cvs!["unix"],
            os: "nuttx".into(),
            abi: "eabi".into(),
            llvm_floatabi: Some(FloatAbi::Soft),
            // The ARMv6-M architecture doesn't support unaligned loads/stores so we disable them
            // with +strict-align.
            // The ARMv6-M doesn't support hardware atomic operations, use atomic builtins instead.
            features: "+strict-align".into(),
            max_atomic_width: Some(32),
            ..base::thumb::opts()
        },
    }
}
