use crate::abi::Endian;
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "mipsisa32r6-unknown-linux-gnu".into(),
        pointer_width: 32,
        data_layout: "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64".into(),
        arch: "mips32r6".into(),
        options: TargetOptions {
            endian: Endian::Big,
            cpu: "mips32r6".into(),
            features: "+mips32r6".into(),
            max_atomic_width: Some(32),
            mcount: "_mcount".into(),

            ..super::linux_gnu_base::opts()
        },
    }
}
