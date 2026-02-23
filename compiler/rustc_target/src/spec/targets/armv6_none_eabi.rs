//! Targets the ARMv6K architecture, with `a32` code by default.

use crate::spec::{Abi, Arch, FloatAbi, Target, TargetMetadata, TargetOptions, base, cvs};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "armv6-none-eabi".into(),
        metadata: TargetMetadata {
            description: Some("Bare ARMv6 soft-float".into()),
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
            asm_args: cvs!["-mthumb-interwork", "-march=armv6", "-mlittle-endian",],
            features: "+soft-float,+strict-align,+v6k".into(),
            atomic_cas: true,
            has_thumb_interworking: true,
            // LDREXD/STREXD available as of ARMv6K
            max_atomic_width: Some(64),
            ..base::arm_none::opts()
        },
    }
}
