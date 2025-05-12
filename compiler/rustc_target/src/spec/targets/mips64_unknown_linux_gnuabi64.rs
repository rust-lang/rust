use rustc_abi::Endian;

use crate::spec::{Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "mips64-unknown-linux-gnuabi64".into(),
        metadata: TargetMetadata {
            description: Some("MIPS64 Linux, N64 ABI (kernel 4.4, glibc 2.23)".into()),
            tier: Some(3),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "E-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".into(),
        arch: "mips64".into(),
        options: TargetOptions {
            abi: "abi64".into(),
            endian: Endian::Big,
            // NOTE(mips64r2) matches C toolchain
            cpu: "mips64r2".into(),
            features: "+mips64r2,+xgot".into(),
            max_atomic_width: Some(64),
            mcount: "_mcount".into(),
            llvm_abiname: "n64".into(),

            ..base::linux_gnu::opts()
        },
    }
}
