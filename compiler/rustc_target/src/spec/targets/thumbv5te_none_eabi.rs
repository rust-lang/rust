//! Targets the ARMv5TE, with code as `t32` code by default.

use crate::spec::{Abi, Arch, FloatAbi, Target, TargetMetadata, TargetOptions, base, cvs};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "thumbv5te-none-eabi".into(),
        metadata: TargetMetadata {
            description: Some("Thumb-mode Bare ARMv5TE".into()),
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
            asm_args: cvs!["-mthumb-interwork", "-march=armv5te", "-mlittle-endian",],
            features: "+soft-float,+strict-align".into(),
            atomic_cas: false,
            max_atomic_width: Some(0),
            has_thumb_interworking: true,
            ..base::arm_none::opts()
        },
    }
}
