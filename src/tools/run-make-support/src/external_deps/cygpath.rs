use std::panic;
use std::path::Path;

use crate::command::Command;
use crate::util::handle_failed_output;

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
pub fn get_windows_path<P: AsRef<Path>>(path: P) -> String {
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
