// Targets the Cortex-M0, Cortex-M0+ and Cortex-M1 processors (ARMv6-M architecture)

use crate::spec::{LinkerFlavor, LldFlavor, Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "thumbv6m-none-eabi".to_string(),
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
            // The ARMv6-M architecture doesn't support unaligned loads/stores so we disable them
            // with +strict-align.
            features: "+strict-align".to_string(),
            // There are no atomic CAS instructions available in the instruction set of the ARMv6-M
            // architecture
            atomic_cas: false,
            ..super::thumb_base::opts()
        },
    }
}
