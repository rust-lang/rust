//! Targets the ARMv4T, with code as `a32` code by default.
//!
//! Primarily of use for the GBA, but usable with other devices too.
//!
//! Please ping @Lokathor if changes are needed.
//!
//! **Important:** This target profile **does not** specify a linker script. You
//! just get the default link script when you build a binary for this target.
//! The default link script is very likely wrong, so you should use
//! `-Clink-arg=-Tmy_script.ld` to override that with a correct linker script.

use crate::spec::{Abi, Arch, FloatAbi, Target, TargetMetadata, TargetOptions, base, cvs};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "armv4t-none-eabi".into(),
        metadata: TargetMetadata {
            description: Some("Bare Armv4T".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        arch: Arch::Arm,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        options: TargetOptions {
            abi: Abi::Eabi,
            llvm_floatabi: Some(FloatAbi::Soft),
            asm_args: cvs!["-mthumb-interwork", "-march=armv4t", "-mlittle-endian",],
            features: "+soft-float,+strict-align".into(),
            atomic_cas: false,
            max_atomic_width: Some(0),
            has_thumb_interworking: true,
            ..base::arm_none::opts()
        },
    }
}
