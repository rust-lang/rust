use crate::spec::{LinkerFlavor, LldFlavor, Target};

pub fn target() -> Target {
    let mut base = super::windows_uwp_gnu_base::opts();
    base.cpu = "x86-64".to_string();
    let gcc_pre_link_args = base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap();
    gcc_pre_link_args.push("-m64".to_string());
    // Use high-entropy 64 bit address space for ASLR
    gcc_pre_link_args.push("-Wl,--high-entropy-va".to_string());
    base.pre_link_args
        .insert(LinkerFlavor::Lld(LldFlavor::Ld), vec!["-m".to_string(), "i386pep".to_string()]);
    base.max_atomic_width = Some(64);

    Target {
        llvm_target: "x86_64-pc-windows-gnu".to_string(),
        target_endian: "little".to_string(),
        pointer_width: 64,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
            .to_string(),
        arch: "x86_64".to_string(),
        target_os: "windows".to_string(),
        target_env: "gnu".to_string(),
        target_vendor: "uwp".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: base,
    }
}
