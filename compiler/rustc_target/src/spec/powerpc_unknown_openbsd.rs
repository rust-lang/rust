use crate::abi::Endian;
use crate::spec::Target;

pub fn target() -> Target {
    let mut base = super::openbsd_base::opts();
    base.endian = Endian::Big;
    base.max_atomic_width = Some(32);

    Target {
        llvm_target: "powerpc-unknown-openbsd".to_string(),
        pointer_width: 32,
        data_layout: "E-m:e-p:32:32-i64:64-n32".to_string(),
        arch: "powerpc".to_string(),
        options: base,
    }
}
