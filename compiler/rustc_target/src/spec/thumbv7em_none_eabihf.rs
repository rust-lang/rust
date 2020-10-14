// Targets the Cortex-M4F and Cortex-M7F processors (ARMv7E-M)
//
// This target assumes that the device does have a FPU (Floating Point Unit) and lowers all (single
// precision) floating point operations to hardware instructions.
//
// Additionally, this target uses the "hard" floating convention (ABI) where floating point values
// are passed to/from subroutines via FPU registers (S0, S1, D0, D1, etc.).
//
// To opt into double precision hardware support, use the `-C target-feature=+fp64` flag.

use crate::spec::{LinkerFlavor, LldFlavor, Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "thumbv7em-none-eabihf".to_string(),
        target_endian: "little".to_string(),
        pointer_width: 32,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        target_os: "none".to_string(),
        target_env: String::new(),
        target_vendor: String::new(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),

        options: TargetOptions {
            // `+vfp4` is the lowest common denominator between the Cortex-M4 (vfp4-16) and the
            // Cortex-M7 (vfp5)
            // `-d32` both the Cortex-M4 and the Cortex-M7 only have 16 double-precision registers
            // available
            // `-fp64` The Cortex-M4 only supports single precision floating point operations
            // whereas in the Cortex-M7 double precision is optional
            //
            // Reference:
            // ARMv7-M Architecture Reference Manual - A2.5 The optional floating-point extension
            features: "+vfp4,-d32,-fp64".to_string(),
            max_atomic_width: Some(32),
            ..super::thumb_base::opts()
        },
    }
}
