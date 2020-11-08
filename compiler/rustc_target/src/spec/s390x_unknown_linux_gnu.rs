use crate::spec::Target;

pub fn target() -> Target {
    let mut base = super::linux_base::opts();
    base.target_endian = "big".to_string();
    // z10 is the oldest CPU supported by LLVM
    base.cpu = "z10".to_string();
    // FIXME: The data_layout string below and the ABI implementation in
    // cabi_s390x.rs are for now hard-coded to assume the no-vector ABI.
    // Pass the -vector feature string to LLVM to respect this assumption.
    base.features = "-vector".to_string();
    base.max_atomic_width = Some(64);
    base.min_global_align = Some(16);

    Target {
        llvm_target: "s390x-unknown-linux-gnu".to_string(),
        pointer_width: 64,
        data_layout: "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-a:8:16-n32:64".to_string(),
        arch: "s390x".to_string(),
        options: base,
    }
}
