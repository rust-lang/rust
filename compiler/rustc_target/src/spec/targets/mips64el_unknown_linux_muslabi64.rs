use crate::spec::{Target, TargetOptions, base};

pub(crate) fn target() -> Target {
    let mut base = base::linux_musl::opts();
    base.cpu = "mips64r2".into();
    base.features = "+mips64r2".into();
    base.max_atomic_width = Some(64);
    Target {
        // LLVM doesn't recognize "muslabi64" yet.
        llvm_target: "mips64el-unknown-linux-musl".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("MIPS64 Linux, N64 ABI, musl 1.2.3".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".into(),
        arch: "mips64".into(),
        options: TargetOptions { abi: "abi64".into(), mcount: "_mcount".into(), ..base },
    }
}
