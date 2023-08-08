use crate::abi::Endian;
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "aarch64_be-unknown-netbsd".into(),
        pointer_width: 64,
        data_layout: "E-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".into(),
        arch: "aarch64".into(),
        options: TargetOptions {
            mcount: "__mcount".into(),
            max_atomic_width: Some(128),
            endian: Endian::Big,
            ..super::netbsd_base::opts()
        },
    }
}
