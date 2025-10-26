use rustc_abi::Endian;

use crate::spec::{Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    let mut base = base::linux_musl::opts();
    base.cpu = "mips64r2".into();
    base.features = "+mips64r2,+xgot".into();
    base.max_atomic_width = Some(64);
    Target {
        // LLVM doesn't recognize "muslabi64" yet.
        llvm_target: "mips64-unknown-linux-musl".into(),
        metadata: TargetMetadata {
            description: Some("MIPS64 Linux, N64 ABI, musl 1.2.3".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "E-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".into(),
        arch: "mips64".into(),
        options: TargetOptions {
            abi: "abi64".into(),
            endian: Endian::Big,
            mcount: "_mcount".into(),
            llvm_abiname: "n64".into(),
            ..base
        },
    }
}
