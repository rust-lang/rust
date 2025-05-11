// Targets Cortex-A7/A8/A9 processors (ARMv7-A)
//
// This target assumes that the device has a FPU (Floating Point Unit) and lowers all (single
// precision) floating point operations to hardware instructions. Cortex-A7/A8/A9 processors
// support VFPv3-D32 or VFPv4-D32 floating point units with optional double-precision support.
//
// This target uses the "hard" floating convention (ABI) where floating point values
// are passed to/from subroutines via FPU registers (S0, S1, D0, D1, etc.).

use crate::spec::{FloatAbi, Target, TargetMetadata, TargetOptions, base, cvs};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "thumbv7a-none-eabihf".into(),
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
            abi: "eabihf".into(),
            llvm_floatabi: Some(FloatAbi::Hard),
            // Cortex-A7/A8/A9 support VFPv3-D32/VFPv4-D32 with optional double-precision
            // and NEON SIMD instructions
            features: "+vfp3,+neon".into(),
            max_atomic_width: Some(64),
            ..base::thumb::opts()
        },
    }
}
