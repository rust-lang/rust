use crate::abi::Endian;
use crate::spec::{LinkerFlavor, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::netbsd_base::opts();
    base.cpu = "v9".into();
    base.pre_link_args.entry(LinkerFlavor::Gcc).or_default().push("-m64".into());
    base.max_atomic_width = Some(64);

    Target {
        llvm_target: "sparc64-unknown-netbsd".into(),
        pointer_width: 64,
        data_layout: "E-m:e-i64:64-n32:64-S128".into(),
        arch: "sparc64".into(),
        options: TargetOptions { endian: Endian::Big, mcount: "__mcount".into(), ..base },
    }
}
