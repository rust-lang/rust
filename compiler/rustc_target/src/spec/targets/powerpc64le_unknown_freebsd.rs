use crate::spec::{base, Cc, LinkerFlavor, Lld, MaybeLazy, StackProbeType, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = base::freebsd::opts();
    base.cpu = "ppc64le".into();
    base.pre_link_args = MaybeLazy::lazy(|| {
        TargetOptions::link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-m64"])
    });
    base.max_atomic_width = Some(64);
    base.stack_probes = StackProbeType::Inline;

    Target {
        llvm_target: "powerpc64le-unknown-freebsd".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 64,
        data_layout: "e-m:e-Fn32-i64:64-n32:64".into(),
        arch: "powerpc64".into(),
        options: TargetOptions { mcount: "_mcount".into(), ..base },
    }
}
