use crate::spec::{LinkerFlavor, Target, TargetOptions, TargetResult};

// This target is for glibc Linux on ARMv7 without NEON or
// thumb-mode. See the thumbv7neon variant for enabling both.

pub fn target() -> TargetResult {
    let base = super::linux_base::opts();
    Ok(Target {
        llvm_target: "armv7-unknown-linux-gnueabihf".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        target_os: "linux".to_string(),
        target_env: "gnu".to_string(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Gcc,

        options: TargetOptions {
            // Info about features at https://wiki.debian.org/ArmHardFloatPort
            features: "+v7,+vfp3,+d16,+thumb2,-neon".to_string(),
            cpu: "generic".to_string(),
            max_atomic_width: Some(64),
            abi_blacklist: super::arm_base::abi_blacklist(),
            target_mcount: "\u{1}__gnu_mcount_nc".to_string(),
            .. base
        }
    })
}
