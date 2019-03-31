use crate::spec::{LinkerFlavor, Target, TargetOptions, TargetResult};

pub fn target() -> TargetResult {
    let mut base = super::cloudabi_base::opts();
    base.cpu = "cortex-a8".to_string();
    base.max_atomic_width = Some(64);
    base.features = "+v7,+vfp3,+neon".to_string();
    base.abi_blacklist = super::arm_base::abi_blacklist();
    base.linker = Some("armv7-unknown-cloudabi-eabihf-cc".to_string());

    Ok(Target {
        llvm_target: "armv7-unknown-cloudabi-eabihf".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        target_os: "cloudabi".to_string(),
        target_env: String::new(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: TargetOptions {
            target_mcount: "\u{1}mcount".to_string(),
            .. base
        },
    })
}
