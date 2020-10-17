use crate::spec::{LinkerFlavor, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::linux_musl_base::opts();
    base.cpu = "mips64r2".to_string();
    base.features = "+mips64r2".to_string();
    base.max_atomic_width = Some(64);
    Target {
        // LLVM doesn't recognize "muslabi64" yet.
        llvm_target: "mips64-unknown-linux-musl".to_string(),
        target_endian: "big".to_string(),
        pointer_width: 64,
        target_c_int_width: "32".to_string(),
        data_layout: "E-m:e-i8:8:32-i16:16:32-i64:64-n32:64-S128".to_string(),
        arch: "mips64".to_string(),
        target_os: "linux".to_string(),
        target_env: "musl".to_string(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: TargetOptions { target_mcount: "_mcount".to_string(), ..base },
    }
}
