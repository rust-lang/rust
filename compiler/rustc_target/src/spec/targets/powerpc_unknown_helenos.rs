use rustc_abi::Endian;

use crate::spec::{Arch, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::helenos::opts();
    base.endian = Endian::Big;
    base.max_atomic_width = Some(32);
    base.linker = Some("ppc-helenos-gcc".into());

    Target {
        llvm_target: "powerpc-unknown-helenos".into(),
        metadata: TargetMetadata {
            description: Some("PowerPC HelenOS".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "E-m:e-p:32:32-Fn32-i64:64-n32".into(),
        arch: Arch::PowerPC,
        options: base,
    }
}
