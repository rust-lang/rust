use crate::spec::{LinkerFlavor, Target};

pub fn target() -> Target {
    let mut base = super::windows_uwp_msvc_base::opts();
    base.cpu = "pentium4".to_string();
    base.max_atomic_width = Some(64);
    base.has_elf_tls = true;

    Target {
        llvm_target: "i686-pc-windows-msvc".to_string(),
        target_endian: "little".to_string(),
        pointer_width: 32,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i64:64-f80:32-n8:16:32-a:0:32-S32"
            .to_string(),
        arch: "x86".to_string(),
        target_os: "windows".to_string(),
        target_env: "msvc".to_string(),
        target_vendor: "uwp".to_string(),
        linker_flavor: LinkerFlavor::Msvc,
        options: base,
    }
}
