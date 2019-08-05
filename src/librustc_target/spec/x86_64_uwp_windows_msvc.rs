use crate::spec::{LinkerFlavor, Target, TargetResult};
use cc::windows_registry;

pub fn target() -> TargetResult {
    let mut base = super::windows_uwp_msvc_base::opts();
    base.cpu = "x86-64".to_string();
    base.max_atomic_width = Some(64);
    base.has_elf_tls = true;

    let link_tool = windows_registry::find_tool("x86_64-pc-windows-msvc", "link.exe")
        .expect("no path found for link.exe");

    let original_path = link_tool.path();
    let lib_root_path = original_path.ancestors().skip(4).next().unwrap().display();

    base.pre_link_args.get_mut(&LinkerFlavor::Msvc).unwrap()
            .push(format!("{}{}{}",
            "/LIBPATH:".to_string(),
            lib_root_path,
            "lib\\x64\\store".to_string()));

    Ok(Target {
        llvm_target: "x86_64-pc-windows-msvc".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "64".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:w-i64:64-f80:128-n8:16:32:64-S128".to_string(),
        arch: "x86_64".to_string(),
        target_os: "windows".to_string(),
        target_env: "msvc".to_string(),
        target_vendor: "uwp".to_string(),
        linker_flavor: LinkerFlavor::Msvc,
        options: base,
    })
}
