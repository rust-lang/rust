use crate::spec::{LinkerFlavor, LldFlavor, Target};

pub fn target() -> Target {
    let mut base = super::hermit_base::opts();
    base.max_atomic_width = Some(128);

    Target {
        llvm_target: "aarch64-unknown-hermit".to_string(),
        target_endian: "little".to_string(),
        pointer_width: 64,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".to_string(),
        arch: "aarch64".to_string(),
        target_os: "hermit".to_string(),
        target_env: String::new(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
        options: base,
    }
}
