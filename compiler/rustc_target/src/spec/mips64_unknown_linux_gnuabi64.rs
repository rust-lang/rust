use crate::abi::Endian;
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "mips64-unknown-linux-gnuabi64".into(),
        pointer_width: 64,
        data_layout: "E-m:e-i8:8:32-i16:16:32-i64:64-n32:64-S128".into(),
        arch: "mips64".into(),
        options: TargetOptions {
            abi: "abi64".into(),
            endian: Endian::Big,
            // NOTE(mips64r2) matches C toolchain
            cpu: "mips64r2".into(),
            features: "+mips64r2,+xgot".into(),
            max_atomic_width: Some(64),
            mcount: "_mcount".into(),

            ..super::linux_gnu_base::opts()
        },
    }
}
