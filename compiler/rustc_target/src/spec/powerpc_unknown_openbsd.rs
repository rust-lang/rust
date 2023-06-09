use crate::abi::Endian;
use crate::spec::{StackProbeType, Target};

pub fn target() -> Target {
    let mut base = super::openbsd_base::opts();
    base.endian = Endian::Big;
    base.max_atomic_width = Some(32);
    base.stack_probes = StackProbeType::Inline;

    Target {
        llvm_target: "powerpc-unknown-openbsd".into(),
        pointer_width: 32,
        data_layout: "E-m:e-p:32:32-i64:64-n32".into(),
        arch: "powerpc".into(),
        options: base,
    }
}
