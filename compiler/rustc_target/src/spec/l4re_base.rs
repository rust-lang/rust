use crate::spec::{LinkArgs, LinkerFlavor, PanicStrategy, TargetOptions};
//use std::process::Command;

// Use GCC to locate code for crt* libraries from the host, not from L4Re. Note
// that a few files also come from L4Re, for these, the function shouldn't be
// used. This uses GCC for the location of the file, but GCC is required for L4Re anyway.
//fn get_path_or(filename: &str) -> String {
//    let child = Command::new("gcc")
//        .arg(format!("-print-file-name={}", filename)).output()
//        .expect("Failed to execute GCC");
//    String::from_utf8(child.stdout)
//        .expect("Couldn't read path from GCC").trim().into()
//}

pub fn opts() -> TargetOptions {
    let mut args = LinkArgs::new();
    args.insert(LinkerFlavor::Gcc, vec![]);

    TargetOptions {
        target_os: "l4re".to_string(),
        target_env: "uclibc".to_string(),
        linker_flavor: LinkerFlavor::Ld,
        executables: true,
        has_elf_tls: false,
        panic_strategy: PanicStrategy::Abort,
        linker: Some("ld".to_string()),
        pre_link_args: args,
        target_family: Some("unix".to_string()),
        ..Default::default()
    }
}
