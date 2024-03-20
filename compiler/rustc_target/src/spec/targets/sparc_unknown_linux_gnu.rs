use crate::abi::Endian;
use crate::spec::{base, Cc, LinkerFlavor, Lld, MaybeLazy, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = base::linux_gnu::opts();
    base.endian = Endian::Big;
    base.cpu = "v9".into();
    base.max_atomic_width = Some(32);
    base.pre_link_args = MaybeLazy::lazy(|| {
        TargetOptions::link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-mv8plus"])
    });

    Target {
        llvm_target: "sparc-unknown-linux-gnu".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 32,
        data_layout: "E-m:e-p:32:32-i64:64-f128:64-n32-S64".into(),
        arch: "sparc".into(),
        options: base,
    }
}
