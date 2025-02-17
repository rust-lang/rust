use crate::spec::{Cc, LinkerFlavor, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::aix::opts();
    base.max_atomic_width = Some(64);
    base.add_pre_link_args(
        LinkerFlavor::Unix(Cc::No),
        &["-b64", "-bpT:0x100000000", "-bpD:0x110000000", "-bcdtors:all:0:s"],
    );

    Target {
        llvm_target: "powerpc64-ibm-aix".into(),
        metadata: TargetMetadata {
            description: Some("64-bit AIX (7.2 and newer)".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: None, // ?
        },
        pointer_width: 64,
        data_layout: "E-m:a-Fi64-i64:64-i128:128-n32:64-S128-v256:256:256-v512:512:512".into(),
        arch: "powerpc64".into(),
        options: base,
    }
}
