use std::{env, path};

use rustc_windows_rc::{VersionInfoFileType, compile_windows_resource_file};

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS");
    let target_env = env::var("CARGO_CFG_TARGET_ENV");
    if Ok("windows") == target_os.as_deref() && Ok("msvc") == target_env.as_deref() {
        set_windows_dll_options();
    } else {
        // Avoid rerunning the build script every time.
        println!("cargo:rerun-if-changed=build.rs");
    }
}

fn set_windows_dll_options() {
    let stem = path::PathBuf::from("rustc_driver_resource");
    let file_description = "rustc_driver";
    let res_file = compile_windows_resource_file(&stem, file_description, VersionInfoFileType::Dll);
    println!("cargo:rustc-link-arg={}", res_file.display());
}
