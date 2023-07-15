//! Some stuff used by rustc that doesn't have many dependencies
//!
//! Originally extracted from rustc::back, which was nominally the
//! compiler 'backend', though LLVM is rustc's backend, so rustc_target
//! is really just odds-and-ends relating to code gen and linking.
//! This crate mostly exists to make rustc smaller, so we might put
//! more 'stuff' here in the future. It does not have a dependency on
//! LLVM.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(assert_matches)]
#![feature(associated_type_bounds)]
#![feature(exhaustive_patterns)]
#![feature(iter_intersperse)]
#![feature(let_chains)]
#![feature(min_specialization)]
#![feature(never_type)]
#![feature(rustc_attrs)]
#![feature(step_trait)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

use std::path::{Path, PathBuf};

#[macro_use]
extern crate rustc_macros;

#[macro_use]
extern crate tracing;

pub mod abi;
pub mod asm;
pub mod json;
pub mod spec;

#[cfg(test)]
mod tests;

pub use rustc_abi::HashStableContext;

/// The name of rustc's own place to organize libraries.
///
/// Used to be `rustc`, now the default is `rustlib`.
const RUST_LIB_DIR: &str = "rustlib";

/// Returns a `rustlib` path for this particular target, relative to the provided sysroot.
///
/// For example: `target_sysroot_path("/usr", "x86_64-unknown-linux-gnu")` =>
/// `"lib*/rustlib/x86_64-unknown-linux-gnu"`.
pub fn target_rustlib_path(sysroot: &Path, target_triple: &str) -> PathBuf {
    let libdir = find_libdir(sysroot);
    PathBuf::from_iter([
        Path::new(libdir.as_ref()),
        Path::new(RUST_LIB_DIR),
        Path::new(target_triple),
    ])
}

/// The name of the directory rustc expects libraries to be located.
fn find_libdir(sysroot: &Path) -> std::borrow::Cow<'static, str> {
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
