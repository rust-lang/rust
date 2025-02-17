// Targets the Cortex-M4 and Cortex-M7 processors (ARMv7E-M)
//
// This target assumes that the device doesn't have a FPU (Floating Point Unit) and lowers all the
// floating point operations to software routines (intrinsics).
//
// As such, this target uses the "soft" calling convention (ABI) where floating point values are
// passed to/from subroutines via general purpose registers (R0, R1, etc.).
//
// To opt-in to hardware accelerated floating point operations, you can use, for example,
// `-C target-feature=+vfp4` or `-C target-cpu=cortex-m4`.

use crate::spec::{FloatAbi, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "thumbv7em-none-eabi".into(),
        metadata: TargetMetadata {
            description: Some("Bare ARMv7E-M".into()),
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
