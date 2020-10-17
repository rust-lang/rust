use crate::spec::{LinkerFlavor, Target};

pub fn target() -> Target {
    let mut base = super::windows_uwp_msvc_base::opts();
    base.max_atomic_width = Some(64);
    base.has_elf_tls = true;

    Target {
        llvm_target: "aarch64-pc-windows-msvc".to_string(),
        target_endian: "little".to_string(),
        pointer_width: 64,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128".to_string(),
        arch: "aarch64".to_string(),
        target_os: "windows".to_string(),
        target_env: "msvc".to_string(),
        target_vendor: "uwp".to_string(),
        linker_flavor: LinkerFlavor::Msvc,
        options: base,
    }
}
