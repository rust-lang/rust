use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::linux_musl_base::opts();
    base.cpu = "mips32r2".to_string();
    base.features = "+mips32r2,+soft-float".to_string();
    base.max_atomic_width = Some(32);
    base.crt_static_default = false;
    Target {
        llvm_target: "mipsel-unknown-linux-musl".to_string(),
        pointer_width: 32,
        data_layout: "e-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64".to_string(),
        arch: "mips".to_string(),
        options: TargetOptions { mcount: "_mcount".to_string(), ..base },
    }
}
