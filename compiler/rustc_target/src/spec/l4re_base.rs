use crate::spec::{LinkerFlavor, PanicStrategy, TargetOptions};
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
    TargetOptions {
        os: "l4re".to_string(),
        env: "uclibc".to_string(),
        linker_flavor: LinkerFlavor::Ld,
        executables: true,
        panic_strategy: PanicStrategy::Abort,
        linker: Some("ld".to_string()),
        linker_is_gnu: false,
        families: vec!["unix".to_string()],
        ..Default::default()
    }
}
