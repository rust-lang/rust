// Targets the Cortex-M55/-M85 processors (Armv8.1-M Mainline architecture profile),
// with the Floating Point extension.
//
// It does not assume you have M-Profile Vector Extensions (aka Helium)

use crate::spec::{Arch, CfgAbi, FloatAbi, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "thumbv8.1m.main-none-eabihf".into(),
        metadata: TargetMetadata {
            description: Some("Bare ARMv8.1-M Mainline, hardfloat, with DSP and LOB".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: Arch::Arm,

        options: TargetOptions {
            cfg_abi: CfgAbi::EabiHf,
            llvm_floatabi: Some(FloatAbi::Hard),
            // The Cortex-M55 and Cortex-M85 both have FPv5-D16 double-precision FPUs
            // We also assume you have the DSP and LOB extensions (which are technically optional).
            features: "+dsp,+fp-armv8d16,+lob".into(),
            max_atomic_width: Some(32),
            ..base::arm_none::opts()
        },
    }
}
