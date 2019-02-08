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

use crate::spec::{LinkerFlavor, LldFlavor, Target, TargetOptions, TargetResult};

pub fn target() -> TargetResult {
    Ok(Target {
        llvm_target: "thumbv7em-none-eabi".to_string(),
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
            max_atomic_width: Some(32),
            .. super::thumb_base::opts()
        },
    })
}
