//! A proc macro that sabotages the incremental compilation finalize step.
//!
//! When invoked, it locates the `-working` session directory inside the
//! incremental compilation directory (passed via POISON_INCR_DIR) and
//! makes it impossible to rename:
//!
//!   - On Unix: removes write permission from the parent (crate) directory.
//!   - On Windows: creates a file inside the -working directory and leaks
//!     the file handle, preventing the directory from being renamed.

extern crate proc_macro;

use std::fs;
use std::path::PathBuf;

use proc_macro::TokenStream;

#[proc_macro]
pub fn poison_finalize(_input: TokenStream) -> TokenStream {
    let incr_dir = std::env::var("POISON_INCR_DIR").expect("POISON_INCR_DIR must be set");

    let crate_dir = find_crate_dir(&incr_dir);
    let working_dir = find_working_dir(&crate_dir);

    #[cfg(unix)]
    poison_unix(&crate_dir);

    #[cfg(windows)]
    poison_windows(&working_dir);

    TokenStream::new()
}

/// Remove write permission from the crate directory.
/// This causes rename() to fail with EACCES
#[cfg(unix)]
fn poison_unix(crate_dir: &PathBuf) {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = fs::metadata(crate_dir).unwrap().permissions();
    perms.set_mode(0o555); // r-xr-xr-x
    fs::set_permissions(crate_dir, perms).unwrap();
}

/// Create a file inside the -working directory and leak the
/// handle. Windows prevents renaming a directory when any file inside it
/// has an open handle. The handle stays open until the rustc process exits.
#[cfg(windows)]
fn poison_windows(working_dir: &PathBuf) {
    let poison_file = working_dir.join("_poison_handle");
    let f = fs::File::create(&poison_file).unwrap();
    // Leak the handle so it stays open for the lifetime of the rustc process.
    std::mem::forget(f);
}

/// Find the crate directory for `foo` inside the incremental compilation dir.
///
/// The incremental directory layout is:
///   {incr_dir}/{crate_name}-{stable_crate_id}/
fn find_crate_dir(incr_dir: &str) -> PathBuf {
    let mut dirs = fs::read_dir(incr_dir).unwrap().filter_map(|e| {
        let e = e.ok()?;
        let name = e.file_name();
        let name = name.to_str()?;
        if e.file_type().ok()?.is_dir() && name.starts_with("foo-") { Some(e.path()) } else { None }
    });

    let first =
        dirs.next().unwrap_or_else(|| panic!("no foo-* crate directory found in {incr_dir}"));
    assert!(
        dirs.next().is_none(),
        "expected exactly one foo-* crate directory in {incr_dir}, found multiple"
    );
    first
}

/// Find the session directory ending in "-working" inside the crate directory
fn find_working_dir(crate_dir: &PathBuf) -> PathBuf {
    for entry in fs::read_dir(crate_dir).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        let name = name.to_str().unwrap().to_string();
        if name.starts_with("s-") && name.ends_with("-working") {
            return entry.path();
        }
    }
    panic!("no -working session directory found in {}", crate_dir.display());
}
