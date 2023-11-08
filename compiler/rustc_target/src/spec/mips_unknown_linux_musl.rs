use crate::abi::Endian;
use crate::spec::{base, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = base::linux_musl::opts();
    base.cpu = "mips32r2".into();
    base.features = "+mips32r2,+soft-float".into();
    base.max_atomic_width = Some(32);
    base.crt_static_default = false;
    Target {
        llvm_target: "mips-unknown-linux-musl".into(),
        pointer_width: 32,
        data_layout: "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64".into(),
        arch: "mips".into(),
        options: TargetOptions { endian: Endian::Big, mcount: "_mcount".into(), ..base },
    }
}
