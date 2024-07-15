//! Collection of path-related helpers.

use std::panic;
use std::path::{Path, PathBuf};

use crate::command::Command;
use crate::env_checked::env_var;
use crate::handle_failed_output;

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

/// Use `cygpath -w` on a path to get a Windows path string back. This assumes that `cygpath` is
/// available on the platform!
///
/// # FIXME
///
/// FIXME(jieyouxu): we should consider not depending on `cygpath`.
///
/// > The cygpath program is a utility that converts Windows native filenames to Cygwin POSIX-style
/// > pathnames and vice versa.
/// >
/// > [irrelevant entries omitted...]
/// >
/// > `-w, --windows         print Windows form of NAMEs (C:\WINNT)`
/// >
/// > -- *from [cygpath documentation](https://cygwin.com/cygwin-ug-net/cygpath.html)*.
#[track_caller]
#[must_use]
pub fn cygpath_windows<P: AsRef<Path>>(path: P) -> String {
    let caller = panic::Location::caller();
    let mut cygpath = Command::new("cygpath");
    cygpath.arg("-w");
    cygpath.arg(path.as_ref());
    let output = cygpath.run();
    if !output.status().success() {
        handle_failed_output(&cygpath, output, caller.line());
    }
    // cygpath -w can attach a newline
    output.stdout_utf8().trim().to_string()
}
