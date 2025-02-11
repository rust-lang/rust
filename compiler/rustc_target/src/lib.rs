//! Some stuff used by rustc that doesn't have many dependencies
//!
//! Originally extracted from rustc::back, which was nominally the
//! compiler 'backend', though LLVM is rustc's backend, so rustc_target
//! is really just odds-and-ends relating to code gen and linking.
//! This crate mostly exists to make rustc smaller, so we might put
//! more 'stuff' here in the future. It does not have a dependency on
//! LLVM.

// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(assert_matches)]
#![feature(debug_closure_helpers)]
#![feature(iter_intersperse)]
#![feature(let_chains)]
#![feature(rustc_attrs)]
#![feature(rustdoc_internals)]
// tidy-alphabetical-end

use std::path::{Path, PathBuf};

pub mod asm;
pub mod callconv;
pub mod json;
pub mod spec;
pub mod target_features;

#[cfg(test)]
mod tests;

use rustc_abi::HashStableContext;

/// The name of rustc's own place to organize libraries.
///
/// Used to be `rustc`, now the default is `rustlib`.
const RUST_LIB_DIR: &str = "rustlib";

/// Returns a `rustlib` path for this particular target, relative to the provided sysroot.
///
/// For example: `target_sysroot_path("/usr", "x86_64-unknown-linux-gnu")` =>
/// `"lib*/rustlib/x86_64-unknown-linux-gnu"`.
pub fn relative_target_rustlib_path(sysroot: &Path, target_triple: &str) -> PathBuf {
    let libdir = find_relative_libdir(sysroot);
    Path::new(libdir.as_ref()).join(RUST_LIB_DIR).join(target_triple)
}

/// The name of the directory rustc expects libraries to be located.
fn find_relative_libdir(sysroot: &Path) -> std::borrow::Cow<'static, str> {
    // FIXME: This is a quick hack to make the rustc binary able to locate
    // Rust libraries in Linux environments where libraries might be installed
    // to lib64/lib32. This would be more foolproof by basing the sysroot off
    // of the directory where `librustc_driver` is located, rather than
    // where the rustc binary is.
    // If --libdir is set during configuration to the value other than
    // "lib" (i.e., non-default), this value is used (see issue #16552).

    #[cfg(target_pointer_width = "64")]
    const PRIMARY_LIB_DIR: &str = "lib64";

    #[cfg(target_pointer_width = "32")]
    const PRIMARY_LIB_DIR: &str = "lib32";

    const SECONDARY_LIB_DIR: &str = "lib";

    match option_env!("CFG_LIBDIR_RELATIVE") {
        None | Some("lib") => {
            if sysroot.join(PRIMARY_LIB_DIR).join(RUST_LIB_DIR).exists() {
                PRIMARY_LIB_DIR.into()
            } else {
                SECONDARY_LIB_DIR.into()
            }
        }
        Some(libdir) => libdir.into(),
    }
}
