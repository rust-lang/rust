use rustc_abi::Endian;

use crate::spec::{Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    let mut base = base::linux_musl::opts();
    base.cpu = "mips32r2".into();
    base.features = "+mips32r2,+soft-float".into();
    base.max_atomic_width = Some(32);
    Target {
        llvm_target: "mips-unknown-linux-musl".into(),
        metadata: TargetMetadata {
            description: Some("MIPS Linux with musl 1.2.3".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64".into(),
        arch: "mips".into(),
        options: TargetOptions { endian: Endian::Big, mcount: "_mcount".into(), ..base },
    }
}
