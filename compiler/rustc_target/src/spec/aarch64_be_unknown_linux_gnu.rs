use crate::abi::Endian;
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "aarch64_be-unknown-linux-gnu".into(),
        pointer_width: 64,
        data_layout: "E-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".into(),
        arch: "aarch64".into(),
        options: TargetOptions {
            features: "+v8a,+outline-atomics".into(),
            max_atomic_width: Some(128),
            mcount: "\u{1}_mcount".into(),
            endian: Endian::Big,
            ..super::linux_gnu_base::opts()
        },
    }
}
