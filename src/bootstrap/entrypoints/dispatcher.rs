//! rustbuild, the Rust build system
//!
//! This is the entry point for the build system used to compile the `rustc`
//! compiler. Lots of documentation can be found in the `README.md` file in the
//! parent directory, and otherwise documentation can be found throughout the `build`
//! directory in each respective module.

use std::env;
use std::env::consts::EXE_SUFFIX;
use std::ffi::OsStr;
use std::path::Path;

mod bootstrap;
mod llvm_config_wrapper;
mod rustc;
mod rustdoc;
mod sccache_plus_cl;

fn main() {
    match env::args_os()
        .next()
        .as_deref()
        .map(Path::new)
        .and_then(Path::file_name)
        .and_then(OsStr::to_str)
        .map(|s| s.strip_suffix(EXE_SUFFIX).unwrap_or(s))
    {
        // the shim name is here to make default-run work.
        Some("bootstrap" | "rustbuild-binary-dispatch-shim") => bootstrap::main(),
        Some("rustc") => rustc::main(),
        Some("rustdoc") => rustdoc::main(),
        Some("sccache-plus-cl") => sccache_plus_cl::main(),
        Some("llvm-config-wrapper") => llvm_config_wrapper::main(),
        Some(arg) => panic!("invalid executable name: {}", arg),
        None => panic!("argv[0] does not exist"),
    }
}
