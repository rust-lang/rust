use crate::spec::{Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "mipsisa64r6el-unknown-linux-gnuabi64".into(),
        metadata: TargetMetadata {
            description: Some("64-bit MIPS Release 6 Little Endian".into()),
            tier: Some(3),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".into(),
        arch: "mips64r6".into(),
        options: TargetOptions {
            abi: "abi64".into(),
            // NOTE(mips64r6) matches C toolchain
            cpu: "mips64r6".into(),
            features: "+mips64r6".into(),
            max_atomic_width: Some(64),
            mcount: "_mcount".into(),
            llvm_abiname: "n64".into(),

            ..base::linux_gnu::opts()
        },
    }
}
