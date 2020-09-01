use crate::spec::{LinkerFlavor, Target, TargetResult};

pub fn target() -> TargetResult {
    let mut base = super::linux_base::opts();
    // z10 is the oldest CPU supported by LLVM
    base.cpu = "z10".to_string();
    // FIXME: The data_layout string below and the ABI implementation in
    // cabi_s390x.rs are for now hard-coded to assume the no-vector ABI.
    // Pass the -vector feature string to LLVM to respect this assumption.
    base.features = "-vector".to_string();
    base.max_atomic_width = Some(64);
    base.min_global_align = Some(16);

    Ok(Target {
        llvm_target: "s390x-unknown-linux-gnu".to_string(),
        target_endian: "big".to_string(),
        target_pointer_width: "64".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-a:8:16-n32:64".to_string(),
        arch: "s390x".to_string(),
        target_os: "linux".to_string(),
        target_env: "gnu".to_string(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: base,
    })
}
