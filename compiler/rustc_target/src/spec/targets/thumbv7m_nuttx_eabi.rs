// Targets the Cortex-M3 processor (ARMv7-M)

use crate::spec::{Abi, Arch, FloatAbi, Os, Target, TargetMetadata, TargetOptions, base, cvs};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "thumbv7m-none-eabi".into(),
        metadata: TargetMetadata {
            description: None,
            tier: Some(3),
            host_tools: None,
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: Arch::Arm,

        options: TargetOptions {
            families: cvs!["unix"],
            os: Os::NuttX,
            abi: Abi::Eabi,
            llvm_floatabi: Some(FloatAbi::Soft),
            max_atomic_width: Some(32),
            ..base::arm_none::opts()
        },
    }
}
