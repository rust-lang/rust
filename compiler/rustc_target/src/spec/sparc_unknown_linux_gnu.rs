use crate::abi::Endian;
use crate::spec::{base, Cc, LinkerFlavor, Lld, Target};

pub fn target() -> Target {
    let mut base = base::linux_gnu::opts();
    base.endian = Endian::Big;
    base.cpu = "v9".into();
    base.max_atomic_width = Some(32);
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-mv8plus"]);

    Target {
        llvm_target: "sparc-unknown-linux-gnu".into(),
        pointer_width: 32,
        data_layout: "E-m:e-p:32:32-i64:64-f128:64-n32-S64".into(),
        arch: "sparc".into(),
        options: base,
    }
}
