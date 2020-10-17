use crate::spec::{LinkerFlavor, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::linux_musl_base::opts();

    // Most of these settings are copied from the arm_unknown_linux_gnueabi
    // target.
    base.features = "+strict-align,+v6".to_string();
    base.max_atomic_width = Some(64);
    Target {
        // It's important we use "gnueabi" and not "musleabi" here. LLVM uses it
        // to determine the calling convention and float ABI, and it doesn't
        // support the "musleabi" value.
        llvm_target: "arm-unknown-linux-gnueabi".to_string(),
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
            unsupported_abis: super::arm_base::unsupported_abis(),
            target_mcount: "\u{1}mcount".to_string(),
            ..base
        },
    }
}
