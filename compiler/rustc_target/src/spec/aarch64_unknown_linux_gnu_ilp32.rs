use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "aarch64-unknown-linux-gnu_ilp32".into(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".into(),
        arch: "aarch64".into(),
        options: TargetOptions {
            abi: "ilp32".into(),
            features: "+v8a,+outline-atomics".into(),
            max_atomic_width: Some(128),
            mcount: "\u{1}_mcount".into(),
            ..super::linux_gnu_base::opts()
        },
    }
}
