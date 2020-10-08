use crate::spec::Target;

pub fn target() -> Target {
    let mut base = super::linux_base::opts();
    base.target_endian = "big".to_string();
    base.cpu = "v9".to_string();
    base.max_atomic_width = Some(64);

    Target {
        llvm_target: "sparc64-unknown-linux-gnu".to_string(),
        pointer_width: 64,
        data_layout: "E-m:e-i64:64-n32:64-S128".to_string(),
        arch: "sparc64".to_string(),
        options: base,
    }
}
