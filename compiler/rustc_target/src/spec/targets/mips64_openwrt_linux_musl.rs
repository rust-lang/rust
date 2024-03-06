/// A target tuple for OpenWrt MIPS64 targets
///
use crate::abi::Endian;
use crate::spec::{base, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = base::linux_musl::opts();
    base.cpu = "mips64r2".into();
    base.features = "+mips64r2,+soft-float".into();
    base.max_atomic_width = Some(64);
    base.crt_static_default = false;

    Target {
        // LLVM doesn't recognize "muslabi64" yet.
        llvm_target: "mips64-unknown-linux-musl".into(),
        description: None,
        pointer_width: 64,
        data_layout: "E-m:e-i8:8:32-i16:16:32-i64:64-n32:64-S128".into(),
        arch: "mips64".into(),
        options: TargetOptions {
            abi: "abi64".into(),
            endian: Endian::Big,
            mcount: "_mcount".into(),
            ..base
        },
    }
}
