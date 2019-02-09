use crate::spec::{LinkerFlavor, Target, TargetOptions, TargetResult};
use super::apple_ios_base::{opts, Arch};

pub fn target() -> TargetResult {
    let base = opts(Arch::Armv7s)?;
    Ok(Target {
        llvm_target: "armv7s-apple-ios".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32".to_string(),
        arch: "arm".to_string(),
        target_os: "ios".to_string(),
        target_env: String::new(),
        target_vendor: "apple".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: TargetOptions {
            features: "+v7,+vfp4,+neon".to_string(),
            max_atomic_width: Some(64),
            abi_blacklist: super::arm_base::abi_blacklist(),
            .. base
        }
    })
}
