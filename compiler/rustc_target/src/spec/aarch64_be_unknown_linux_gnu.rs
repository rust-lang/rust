use crate::abi::Endian;
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "aarch64_be-unknown-linux-gnu".to_string(),
        pointer_width: 64,
        data_layout: "E-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".to_string(),
        arch: "aarch64".to_string(),
        options: TargetOptions {
            features: "+outline-atomics".to_string(),
            max_atomic_width: Some(128),
            mcount: "\u{1}_mcount".to_string(),
            endian: Endian::Big,
            ..super::linux_gnu_base::opts()
        },
    }
}
