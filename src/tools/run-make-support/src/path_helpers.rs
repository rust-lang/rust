//! Collection of path-related helpers.

use std::path::{Path, PathBuf};

use crate::env::env_var;

/// Return the current working directory.
///
/// This forwards to [`std::env::current_dir`], please see its docs regarding platform-specific
/// behavior.
#[must_use]
pub fn cwd() -> PathBuf {
    std::env::current_dir().unwrap()
}

/// Construct a `PathBuf` relative to the current working directory by joining `cwd()` with the
/// relative path. This is mostly a convenience helper so the test writer does not need to write
/// `PathBuf::from(path_like_string)`.
///
/// # Example
///
/// ```rust
/// let p = path("support_file.txt");
/// ```
pub fn path<P: AsRef<Path>>(p: P) -> PathBuf {
    cwd().join(p.as_ref())
}

/// Path to the root `rust-lang/rust` source checkout.
#[must_use]
pub fn source_root() -> PathBuf {
    env_var("SOURCE_ROOT").into()
}
