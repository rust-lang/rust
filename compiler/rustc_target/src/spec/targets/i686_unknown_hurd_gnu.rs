use crate::spec::{base, Cc, LinkerFlavor, Lld, MaybeLazy, StackProbeType, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = base::hurd_gnu::opts();
    base.cpu = "pentiumpro".into();
    base.max_atomic_width = Some(64);
    base.pre_link_args = MaybeLazy::lazy(|| {
        TargetOptions::link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-m32"])
    });
    base.stack_probes = StackProbeType::Inline;

    Target {
        llvm_target: "i686-unknown-hurd-gnu".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i128:128-f64:32:64-f80:32-n8:16:32-S128"
            .into(),
        arch: "x86".into(),
        options: base,
    }
}
