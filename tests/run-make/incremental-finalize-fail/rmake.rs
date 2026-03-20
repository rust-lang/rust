//! Test that a failure to finalize the incremental compilation session directory
//! (i.e., the rename from "-working" to the SVH-based name) results in a
//! note, not an ICE, and that the compilation output is still produced.
//!
//! Strategy:
//!   1. Build the `poison` proc-macro crate
//!   2. Compile foo.rs with incremental compilation
//!      The proc macro runs mid-compilation (after prepare_session_directory
//!      but before finalize_session_directory) and sabotages the rename:
//!        - On Unix: removes write permission from the crate directory,
//!          so rename() fails with EACCES.
//!        - On Windows: creates and leaks an open file handle inside the
//!          -working directory, so rename() fails with ERROR_ACCESS_DENIED.
//!   3. Assert that stderr contains the finalize failure messages

use std::fs;
use std::path::{Path, PathBuf};

use run_make_support::rustc;

/// Guard that restores permissions on the incremental directory on drop,
/// to ensure cleanup is possible
struct IncrDirCleanup;

fn main() {
    let _cleanup = IncrDirCleanup;

    // Build the poison proc-macro crate
    rustc().input("poison/lib.rs").crate_name("poison").crate_type("proc-macro").run();

    let poison_dylib = find_proc_macro_dylib("poison");

    // Incremental compile with the poison macro active
    let out = rustc()
        .input("foo.rs")
        .crate_type("rlib")
        .incremental("incr")
        .extern_("poison", &poison_dylib)
        .env("POISON_INCR_DIR", "incr")
        .run();

    out.assert_stderr_contains("note: did not finalize incremental compilation session directory");
    out.assert_stderr_contains(
        "help: the next build will not be able to reuse work from this compilation",
    );
    out.assert_stderr_not_contains("internal compiler error");
}

impl Drop for IncrDirCleanup {
    fn drop(&mut self) {
        let incr = Path::new("incr");
        if !incr.exists() {
            return;
        }

        #[cfg(unix)]
        restore_permissions(incr);
    }
}

/// Recursively restore write permissions so rm -rf works after the chmod trick
#[cfg(unix)]
fn restore_permissions(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.filter_map(|e| e.ok()) {
            if entry.file_type().map_or(false, |ft| ft.is_dir()) {
                let mut perms = match fs::metadata(entry.path()) {
                    Ok(m) => m.permissions(),
                    Err(_) => continue,
                };
                perms.set_mode(0o755);
                let _ = fs::set_permissions(entry.path(), perms);
            }
        }
    }
}

/// Locate the compiled proc-macro dylib by scanning the current directory.
fn find_proc_macro_dylib(name: &str) -> PathBuf {
    let prefix = if cfg!(target_os = "windows") { "" } else { "lib" };

    let ext: &str = if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else {
        "so"
    };

    let lib_name = format!("{prefix}{name}.{ext}");

    for entry in fs::read_dir(".").unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        let name = name.to_str().unwrap();
        if name == lib_name {
            return entry.path();
        }
    }

    panic!("could not find proc-macro dylib for `{name}`");
}
