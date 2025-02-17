// Targets Cortex-A7/A8/A9 processors (ARMv7-A)
//
// This target assumes that the device does NOT have a FPU (Floating Point Unit)
// and will use software floating point operations. This matches the NuttX EABI
// configuration without hardware floating point support.

use crate::spec::{FloatAbi, Target, TargetMetadata, TargetOptions, base, cvs};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "thumbv7a-none-eabi".into(),
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
            // Cortex-A7/A8/A9 with software floating point
            features: "+soft-float,-neon".into(),
            max_atomic_width: Some(64),
            ..base::thumb::opts()
        },
    }
}
