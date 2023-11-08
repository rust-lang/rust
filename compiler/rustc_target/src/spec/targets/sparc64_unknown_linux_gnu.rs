use crate::abi::Endian;
use crate::spec::{base, Target};

pub fn target() -> Target {
    let mut base = base::linux_gnu::opts();
    base.endian = Endian::Big;
    base.cpu = "v9".into();
    base.max_atomic_width = Some(64);

    Target {
        llvm_target: "sparc64-unknown-linux-gnu".into(),
        pointer_width: 64,
        data_layout: "E-m:e-i64:64-n32:64-S128".into(),
        arch: "sparc64".into(),
        options: base,
    }
}
