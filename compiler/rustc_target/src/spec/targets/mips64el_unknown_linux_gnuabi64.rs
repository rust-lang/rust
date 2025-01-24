use crate::spec::{Target, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "mips64el-unknown-linux-gnuabi64".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("MIPS64 Linux, N64 ABI (kernel 4.4, glibc 2.23)".into()),
            tier: Some(3),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".into(),
        arch: "mips64".into(),
        options: TargetOptions {
            abi: "abi64".into(),
            // NOTE(mips64r2) matches C toolchain
            cpu: "mips64r2".into(),
            features: "+mips64r2,+xgot".into(),
            max_atomic_width: Some(64),
            mcount: "_mcount".into(),

            ..base::linux_gnu::opts()
        },
    }
}
