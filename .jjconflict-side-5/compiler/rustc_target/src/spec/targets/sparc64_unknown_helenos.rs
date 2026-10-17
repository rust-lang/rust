use rustc_abi::Endian;

use crate::spec::{Arch, Cc, LinkerFlavor, Lld, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::helenos::opts();
    base.endian = Endian::Big;
    base.cpu = "v9".into();
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-m64"]);
    base.max_atomic_width = Some(64);
    base.linker = Some("sparc64-helenos-gcc".into());

    Target {
        llvm_target: "sparc64-unknown-helenos".into(),
        metadata: TargetMetadata {
            description: Some("SPARC HelenOS".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "E-m:e-i64:64-i128:128-n32:64-S128".into(),
        arch: Arch::Sparc64,
        options: base,
    }
}
