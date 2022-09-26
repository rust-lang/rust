use crate::abi::Endian;
use crate::spec::{StackProbeType, Target};

pub fn target() -> Target {
    let mut base = super::linux_musl_base::opts();
    base.endian = Endian::Big;
    // z10 is the oldest CPU supported by LLVM
    base.cpu = "z10".into();
    // FIXME: The data_layout string below and the ABI implementation in
    // cabi_s390x.rs are for now hard-coded to assume the no-vector ABI.
    // Pass the -vector feature string to LLVM to respect this assumption.
    base.features = "-vector".into();
    base.max_atomic_width = Some(64);
    base.min_global_align = Some(16);
    base.static_position_independent_executables = true;
    base.stack_probes = StackProbeType::Inline;

    Target {
        llvm_target: "s390x-unknown-linux-musl".into(),
        pointer_width: 64,
        data_layout: "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-a:8:16-n32:64".into(),
        arch: "s390x".into(),
        options: base,
    }
}
