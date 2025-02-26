use rustc_abi::Endian;

use crate::spec::{Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::linux_gnu::opts();
    base.endian = Endian::Big;
    base.cpu = "v9".into();
    base.max_atomic_width = Some(64);

    Target {
        llvm_target: "sparc64-unknown-linux-gnu".into(),
        metadata: TargetMetadata {
            description: Some("SPARC Linux (kernel 4.4, glibc 2.23)".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "E-m:e-i64:64-i128:128-n32:64-S128".into(),
        arch: "sparc64".into(),
        options: base,
    }
}
