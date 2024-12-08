// Targets the Cortex-M0, Cortex-M0+ and Cortex-M1 processors (ARMv6-M architecture)

use crate::spec::{Target, TargetOptions, base, cvs};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "thumbv6m-none-eabi".into(),
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
            abi: "eabi".into(),
            // The ARMv6-M architecture doesn't support unaligned loads/stores so we disable them
            // with +strict-align.
            // Also force-enable 32-bit atomics, which allows the use of atomic load/store only.
            // The resulting atomics are ABI incompatible with atomics backed by libatomic.
            features: "+strict-align,+atomics-32".into(),
            // There are no atomic CAS instructions available in the instruction set of the ARMv6-M
            // architecture
            atomic_cas: false,
            ..base::thumb::opts()
        },
    }
}
