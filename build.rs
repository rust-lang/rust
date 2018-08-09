//! This build script ensures that Clippy is not compiled with an
//! incompatible version of rust. It will panic with a descriptive
//! error message instead.
//!
//! We specifially want to ensure that Clippy is only built with a
//! rustc version that is newer or equal to the one specified in the
//! `min_version.txt` file.
//!
//! `min_version.txt` is in the repo but also in the `.gitignore` to
//! make sure that it is not updated manually by accident. Only CI
//! should update that file.
//!
//! This build script was originally taken from the Rocket web framework:
//! https://github.com/SergioBenitez/Rocket

use std::env;

fn main() {
    // Forward the profile to the main compilation
    println!("cargo:rustc-env=PROFILE={}", env::var("PROFILE").unwrap());
    // Don't rebuild even if nothing changed
    println!("cargo:rerun-if-changed=build.rs");
}
