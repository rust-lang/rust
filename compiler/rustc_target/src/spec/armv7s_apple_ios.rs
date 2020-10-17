use super::apple_sdk_base::{opts, Arch};
use crate::spec::{LinkerFlavor, Target, TargetOptions};

pub fn target() -> Target {
    let base = opts(Arch::Armv7s);
    Target {
        llvm_target: "armv7s-apple-ios".to_string(),
        target_endian: "little".to_string(),
        pointer_width: 32,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32".to_string(),
        arch: "arm".to_string(),
        target_os: "ios".to_string(),
        target_env: String::new(),
        target_vendor: "apple".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: TargetOptions {
            features: "+v7,+vfp4,+neon".to_string(),
            max_atomic_width: Some(64),
            unsupported_abis: super::arm_base::unsupported_abis(),
            ..base
        },
    }
}
