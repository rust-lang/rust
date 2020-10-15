use crate::spec::{LinkerFlavor, Target, TargetOptions};

pub fn target() -> Target {
    let base = super::netbsd_base::opts();
    Target {
        llvm_target: "armv7-unknown-netbsdelf-eabihf".to_string(),
        target_endian: "little".to_string(),
        pointer_width: 32,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        target_os: "netbsd".to_string(),
        target_env: "eabihf".to_string(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Gcc,

        options: TargetOptions {
            features: "+v7,+vfp3,-d32,+thumb2,-neon".to_string(),
            cpu: "generic".to_string(),
            max_atomic_width: Some(64),
            unsupported_abis: super::arm_base::unsupported_abis(),
            target_mcount: "__mcount".to_string(),
            ..base
        },
    }
}
