use crate::spec::{LinkerFlavor, Target, TargetOptions};

// This target is for musl Linux on ARMv7 without thumb-mode, NEON or
// hardfloat.

pub fn target() -> Target {
    let base = super::linux_musl_base::opts();
    // Most of these settings are copied from the armv7_unknown_linux_gnueabi
    // target.
    Target {
        // It's important we use "gnueabi" and not "musleabi" here. LLVM uses it
        // to determine the calling convention and float ABI, and it doesn't
        // support the "musleabi" value.
        llvm_target: "armv7-unknown-linux-gnueabi".to_string(),
        target_endian: "little".to_string(),
        pointer_width: 32,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        target_os: "linux".to_string(),
        target_env: "musl".to_string(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Gcc,

        options: TargetOptions {
            features: "+v7,+thumb2,+soft-float,-neon".to_string(),
            cpu: "generic".to_string(),
            max_atomic_width: Some(64),
            unsupported_abis: super::arm_base::unsupported_abis(),
            target_mcount: "\u{1}mcount".to_string(),
            ..base
        },
    }
}
