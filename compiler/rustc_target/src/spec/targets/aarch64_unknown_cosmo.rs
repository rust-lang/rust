use crate::spec::{Cc, LinkerFlavor, Lld, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::cosmo::opts();

    base.linker = Some("aarch64-unknown-cosmo-cc".into());
    base.features = "+reserve-x28".into();

    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-static"]);

    Target {
        llvm_target: "aarch64-unknown-cosmo".into(),
        metadata: TargetMetadata {
            description: Some("64-bit Cosmopolitan Libc".into()),
            tier: Some(3),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout:
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "aarch64".into(),
        options: base,
    }
}
