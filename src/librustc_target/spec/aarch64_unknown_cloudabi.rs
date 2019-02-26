use crate::spec::{LinkerFlavor, Target, TargetResult};

pub fn target() -> TargetResult {
    let mut base = super::cloudabi_base::opts();
    base.max_atomic_width = Some(128);
    base.abi_blacklist = super::arm_base::abi_blacklist();
    base.linker = Some("aarch64-unknown-cloudabi-cc".to_string());

    Ok(Target {
        llvm_target: "aarch64-unknown-cloudabi".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "64".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".to_string(),
        arch: "aarch64".to_string(),
        target_os: "cloudabi".to_string(),
        target_env: String::new(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: base,
    })
}
