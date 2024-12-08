// Targets the Cortex-M33 processor (Armv8-M Mainline architecture profile),
// with the Floating Point extension.

use crate::spec::{Target, TargetOptions, base, cvs};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "thumbv8m.main-none-eabihf".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),

        options: TargetOptions {
            families: cvs!["unix"],
            os: "nuttx".into(),
            abi: "eabihf".into(),
            // If the Floating Point extension is implemented in the Cortex-M33
            // processor, the Cortex-M33 Technical Reference Manual states that
            // the FPU uses the FPv5 architecture, single-precision instructions
            // and 16 D registers.
            features: "+fp-armv8d16sp".into(),
            max_atomic_width: Some(32),
            ..base::thumb::opts()
        },
    }
}
