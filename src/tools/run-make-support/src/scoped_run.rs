//! Collection of helpers that try to maintain certain properties while running a test closure.

use std::path::Path;

use crate::fs;
use crate::path_helpers::cwd;
use crate::targets::is_windows;

/// Ensure that the path P is read-only while the test runs, and restore original permissions at the
/// end so compiletest can clean up. This will panic on Windows if the path is a directory (as it
/// would otherwise do nothing)
///
/// # Pitfalls
///
/// - Some CI runners are ran as root which may bypass read-only permission restrictions. Unclear
///   exactly when such scenarios occur.
///
/// # FIXME
///
/// FIXME(Oneirical): This will no longer be required after compiletest receives the ability to
/// manipulate read-only files. See <https://github.com/rust-lang/rust/issues/126334>.
#[track_caller]
pub fn test_while_readonly<P, F>(path: P, closure: F)
where
    P: AsRef<Path>,
    F: FnOnce() + std::panic::UnwindSafe,
{
    let path = path.as_ref();
    if is_windows() && path.is_dir() {
        eprintln!("This helper function cannot be used on Windows to make directories readonly.");
        eprintln!(
            "See the official documentation:
            https://doc.rust-lang.org/std/fs/struct.Permissions.html#method.set_readonly"
        );
        panic!("`test_while_readonly` on directory detected while on Windows.");
    }
    let metadata = fs::metadata(&path);
    let original_perms = metadata.permissions();

    let mut new_perms = original_perms.clone();
    new_perms.set_readonly(true);
    fs::set_permissions(&path, new_perms);

    let success = std::panic::catch_unwind(closure);

    fs::set_permissions(&path, original_perms);
    success.unwrap();
}

/// This function is designed for running commands in a temporary directory that is cleared after
/// the function ends.
///
/// What this function does:
/// 1. Creates a temporary directory (`tmpdir`)
/// 2. Copies all files from the current directory to `tmpdir`
/// 3. Changes the current working directory to `tmpdir`
/// 4. Calls `callback`
/// 5. Switches working directory back to the original one
/// 6. Removes `tmpdir`
pub fn run_in_tmpdir<F: FnOnce()>(callback: F) {
    let original_dir = cwd();
    let tmpdir = original_dir.join("../temporary-directory");
    fs::copy_dir_all(".", &tmpdir);

    std::env::set_current_dir(&tmpdir).unwrap();
    callback();
    std::env::set_current_dir(original_dir).unwrap();
    fs::remove_dir_all(tmpdir);
}
