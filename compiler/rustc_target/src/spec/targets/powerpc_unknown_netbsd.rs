use rustc_abi::Endian;

use crate::spec::{
    Cc, LinkerFlavor, Lld, StackProbeType, Target, TargetMetadata, TargetOptions, base,
};

pub(crate) fn target() -> Target {
    let mut base = base::netbsd::opts();
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-m32"]);
    base.max_atomic_width = Some(32);
    base.stack_probes = StackProbeType::Inline;

    Target {
        llvm_target: "powerpc-unknown-netbsd".into(),
        metadata: TargetMetadata {
            description: Some("NetBSD 32-bit powerpc systems".into()),
            tier: Some(3),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "E-m:e-p:32:32-Fn32-i64:64-n32".into(),
        arch: "powerpc".into(),
        options: TargetOptions { endian: Endian::Big, mcount: "__mcount".into(), ..base },
    }
}
