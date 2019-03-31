use crate::spec::{LinkerFlavor, Target, TargetOptions, TargetResult};

// This target is for musl Linux on ARMv7 without thumb-mode or NEON.

pub fn target() -> TargetResult {
    let base = super::linux_musl_base::opts();
    Ok(Target {
        // It's important we use "gnueabihf" and not "musleabihf" here. LLVM
        // uses it to determine the calling convention and float ABI, and LLVM
        // doesn't support the "musleabihf" value.
        llvm_target: "armv7-unknown-linux-gnueabihf".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        target_os: "linux".to_string(),
        target_env: "musl".to_string(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Gcc,

        // Most of these settings are copied from the armv7_unknown_linux_gnueabihf
        // target.
        options: TargetOptions {
            features: "+v7,+vfp3,+d16,+thumb2,-neon".to_string(),
            cpu: "generic".to_string(),
            max_atomic_width: Some(64),
            abi_blacklist: super::arm_base::abi_blacklist(),
            target_mcount: "\u{1}mcount".to_string(),
            .. base
        }
    })
}
