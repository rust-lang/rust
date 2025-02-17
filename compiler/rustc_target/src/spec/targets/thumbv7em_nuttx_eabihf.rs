// Targets the Cortex-M4F and Cortex-M7F processors (ARMv7E-M)
//
// This target assumes that the device does have a FPU (Floating Point Unit) and lowers all (single
// precision) floating point operations to hardware instructions.
//
// Additionally, this target uses the "hard" floating convention (ABI) where floating point values
// are passed to/from subroutines via FPU registers (S0, S1, D0, D1, etc.).
//
// To opt into double precision hardware support, use the `-C target-feature=+fp64` flag.

use crate::spec::{FloatAbi, Target, TargetMetadata, TargetOptions, base, cvs};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "thumbv7em-none-eabihf".into(),
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
            // vfp4 is the lowest common denominator between the Cortex-M4F (vfp4) and the
            // Cortex-M7 (vfp5).
            // Both the Cortex-M4 and the Cortex-M7 only have 16 double-precision registers
            // available, and the Cortex-M4 only supports single-precision floating point operations
            // whereas in the Cortex-M7 double-precision is optional.
            //
            // Reference:
            // ARMv7-M Architecture Reference Manual - A2.5 The optional floating-point extension
            features: "+vfp4d16sp".into(),
            max_atomic_width: Some(32),
            ..base::thumb::opts()
        },
    }
}
