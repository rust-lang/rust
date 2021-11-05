use crate::abi::Endian;
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::linux_gnu_base::opts();
    base.max_atomic_width = Some(128);

    Target {
        llvm_target: "aarch64_be-unknown-linux-gnu_ilp32".to_string(),
        pointer_width: 32,
        data_layout: "E-m:e-p:32:32-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".to_string(),
        arch: "aarch64".to_string(),
        options: TargetOptions {
            abi: "ilp32".to_string(),
            features: "+outline-atomics".to_string(),
            mcount: "\u{1}_mcount".to_string(),
            endian: Endian::Big,
            ..base
        },
    }
}
