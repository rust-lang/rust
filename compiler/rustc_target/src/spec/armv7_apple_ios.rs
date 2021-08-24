use super::apple_sdk_base::{opts, Arch};
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: super::apple_base::ios_llvm_target("armv7"),
        pointer_width: 32,
        data_layout: "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32".to_string(),
        arch: "arm".to_string(),
        options: TargetOptions {
            features: "+v7,+vfp3,+neon".to_string(),
            max_atomic_width: Some(64),
            ..opts("ios", Arch::Armv7)
        },
    }
}
