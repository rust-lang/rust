use crate::spec::{Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "mipsel-unknown-linux-gnu".into(),
        metadata: TargetMetadata {
            description: Some("MIPS (little endian) Linux (kernel 4.4, glibc 2.23)".into()),
            tier: Some(3),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64".into(),
        arch: "mips".into(),

        options: TargetOptions {
            cpu: "mips32r2".into(),
            features: "+mips32r2,+fpxx,+nooddspreg".into(),
            max_atomic_width: Some(32),
            mcount: "_mcount".into(),

            ..base::linux_gnu::opts()
        },
    }
}
