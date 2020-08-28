use crate::spec::{LinkerFlavor, Target, TargetResult};

pub fn target() -> TargetResult {
    let mut base = super::linux_base::opts();
    base.cpu = "v9".to_string();
    base.max_atomic_width = Some(64);

    Ok(Target {
        llvm_target: "sparc64-unknown-linux-gnu".to_string(),
        target_endian: "big".to_string(),
        target_pointer_width: "64".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "E-m:e-i64:64-n32:64-S128".to_string(),
        arch: "sparc64".to_string(),
        target_os: "linux".to_string(),
        target_env: "gnu".to_string(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: base,
    })
}
