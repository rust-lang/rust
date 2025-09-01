use crate::spec::{Cc, LinkerFlavor, Lld, StackProbeType, Target, base};

pub(crate) fn target() -> Target {
    let mut base = base::managarm_mlibc::opts();
    base.cpu = "x86-64".into();
    base.max_atomic_width = Some(64);
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-m64"]);
    base.stack_probes = StackProbeType::Inline;

    Target {
        llvm_target: "x86_64-unknown-managarm-mlibc".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("managarm/amd64".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 64,
        data_layout:
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        options: base,
    }
}
