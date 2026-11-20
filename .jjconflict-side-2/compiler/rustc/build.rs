use std::{env, path};

use rustc_windows_rc::{VersionInfoFileType, compile_windows_resource_file};

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS");
    let target_env = env::var("CARGO_CFG_TARGET_ENV");
    if Ok("windows") == target_os.as_deref() && Ok("msvc") == target_env.as_deref() {
        set_windows_exe_options();
    } else {
        // Avoid rerunning the build script every time.
        println!("cargo:rerun-if-changed=build.rs");
    }
}

// Add a manifest file to rustc.exe.
fn set_windows_exe_options() {
    set_windows_resource();
    set_windows_manifest();
}

fn set_windows_resource() {
    let stem = path::PathBuf::from("rustc_main_resource");
    let file_description = "rustc";
    let res_file = compile_windows_resource_file(&stem, file_description, VersionInfoFileType::App);
    println!("cargo:rustc-link-arg={}", res_file.display());
}

fn set_windows_manifest() {
    static WINDOWS_MANIFEST_FILE: &str = "Windows Manifest.xml";

    let mut manifest = env::current_dir().unwrap();
    manifest.push(WINDOWS_MANIFEST_FILE);

    println!("cargo:rerun-if-changed={WINDOWS_MANIFEST_FILE}");
    // Embed the Windows application manifest file.
    println!("cargo:rustc-link-arg-bin=rustc-main=/MANIFEST:EMBED");
    println!("cargo:rustc-link-arg-bin=rustc-main=/MANIFESTINPUT:{}", manifest.to_str().unwrap());
    // Turn linker warnings into errors.
    println!("cargo:rustc-link-arg-bin=rustc-main=/WX");
}
