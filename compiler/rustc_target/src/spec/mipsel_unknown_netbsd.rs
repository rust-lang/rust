use crate::abi::Endian;
use crate::spec::{base, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = base::netbsd::opts();
    base.max_atomic_width = Some(32);
    base.cpu = "mips32".into();

    Target {
        llvm_target: "mipsel-unknown-netbsd".into(),
        pointer_width: 32,
        data_layout: "e-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64".into(),
        arch: "mips".into(),
        options: TargetOptions {
            features: "+soft-float".into(),
            mcount: "__mcount".into(),
            endian: Endian::Little,
            ..base
        },
    }
}
