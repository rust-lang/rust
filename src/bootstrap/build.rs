use env::consts::{EXE_EXTENSION, EXE_SUFFIX};
use std::env;
use std::ffi::OsString;
use std::path::PathBuf;

/// Given an executable called `name`, return the filename for the
/// executable for a particular target.
pub fn exe(name: &PathBuf) -> PathBuf {
    if EXE_EXTENSION != "" && name.extension() != Some(EXE_EXTENSION.as_ref()) {
        let mut name: OsString = name.clone().into();
        name.push(EXE_SUFFIX);
        name.into()
    } else {
        name.clone()
    }
}

fn main() {
    let host = env::var("HOST").unwrap();
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rustc-env=BUILD_TRIPLE={}", host);

    // This may not be a canonicalized path.
    let mut rustc = PathBuf::from(env::var_os("RUSTC").unwrap());

    if rustc.is_relative() {
        println!("cargo:rerun-if-env-changed=PATH");
        for dir in env::split_paths(&env::var_os("PATH").unwrap_or_default()) {
            let absolute = dir.join(&exe(&rustc));
            if absolute.exists() {
                rustc = absolute;
                break;
            }
        }
    }
    assert!(rustc.is_absolute());

    // FIXME: if the path is not utf-8, this is going to break. Unfortunately
    // Cargo doesn't have a way for us to specify non-utf-8 paths easily, so
    // we'll need to invent some encoding scheme if this becomes a problem.
    println!("cargo:rustc-env=RUSTC={}", rustc.to_str().unwrap());
}
