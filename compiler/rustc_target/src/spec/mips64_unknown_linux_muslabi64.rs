use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::linux_musl_base::opts();
    base.cpu = "mips64r2".to_string();
    base.features = "+mips64r2".to_string();
    base.max_atomic_width = Some(64);
    Target {
        // LLVM doesn't recognize "muslabi64" yet.
        llvm_target: "mips64-unknown-linux-musl".to_string(),
        pointer_width: 64,
        data_layout: "E-m:e-i8:8:32-i16:16:32-i64:64-n32:64-S128".to_string(),
        arch: "mips64".to_string(),
        options: TargetOptions { endian: "big".to_string(), mcount: "_mcount".to_string(), ..base },
    }
}
