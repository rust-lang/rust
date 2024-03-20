use crate::spec::{base, Cc, LinkerFlavor, MaybeLazy, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = base::aix::opts();
    base.max_atomic_width = Some(64);
    base.pre_link_args = MaybeLazy::lazy(|| {
        TargetOptions::link_args(
            LinkerFlavor::Unix(Cc::No),
            &["-b64", "-bpT:0x100000000", "-bpD:0x110000000", "-bcdtors:all:0:s"],
        )
    });

    Target {
        llvm_target: "powerpc64-ibm-aix".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 64,
        data_layout: "E-m:a-Fi64-i64:64-n32:64-S128-v256:256:256-v512:512:512".into(),
        arch: "powerpc64".into(),
        options: base,
    }
}
