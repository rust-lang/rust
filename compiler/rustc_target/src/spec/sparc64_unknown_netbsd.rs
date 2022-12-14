use crate::abi::Endian;
use crate::spec::{Cc, LinkerFlavor, Lld, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::netbsd_base::opts();
    base.cpu = "v9".into();
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-m64"]);
    base.max_atomic_width = Some(64);

    Target {
        llvm_target: "sparc64-unknown-netbsd".into(),
        pointer_width: 64,
        data_layout: "E-m:e-i64:64-n32:64-S128".into(),
        arch: "sparc64".into(),
        options: TargetOptions { endian: Endian::Big, mcount: "__mcount".into(), ..base },
    }
}
