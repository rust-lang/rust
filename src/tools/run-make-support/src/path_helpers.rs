//! Collection of path-related helpers.

use std::path::{Path, PathBuf};

use crate::env::env_var;
use crate::rfs;

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
/// # use run_make_support::path;
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

/// Path to the build directory root.
#[must_use]
pub fn build_root() -> PathBuf {
    env_var("BUILD_ROOT").into()
}

/// Browse the directory `path` non-recursively and return all files which respect the parameters
/// outlined by `closure`.
#[track_caller]
pub fn shallow_find_files<P: AsRef<Path>, F: Fn(&PathBuf) -> bool>(
    path: P,
    filter: F,
) -> Vec<PathBuf> {
    let mut matching_files = Vec::new();
    for entry in rfs::read_dir(path) {
        let entry = entry.expect("failed to read directory entry.");
        let path = entry.path();

        if path.is_file() && filter(&path) {
            matching_files.push(path);
        }
    }
    matching_files
}

/// Browse the directory `path` non-recursively and return all directories which respect the
/// parameters outlined by `closure`.
#[track_caller]
pub fn shallow_find_directories<P: AsRef<Path>, F: Fn(&PathBuf) -> bool>(
    path: P,
    filter: F,
) -> Vec<PathBuf> {
    let mut matching_files = Vec::new();
    for entry in rfs::read_dir(path) {
        let entry = entry.expect("failed to read directory entry.");
        let path = entry.path();

        if path.is_dir() && filter(&path) {
            matching_files.push(path);
        }
    }
    matching_files
}

/// Returns true if the filename at `path` does not contain `expected`.
pub fn not_contains<P: AsRef<Path>>(path: P, expected: &str) -> bool {
    !path.as_ref().file_name().is_some_and(|name| name.to_str().unwrap().contains(expected))
}

/// Returns true if the filename at `path` is not in `expected`.
pub fn filename_not_in_denylist<P: AsRef<Path>, V: AsRef<[String]>>(path: P, expected: V) -> bool {
    let expected = expected.as_ref();
    path.as_ref()
        .file_name()
        .is_some_and(|name| !expected.contains(&name.to_str().unwrap().to_owned()))
}

/// Returns true if the filename at `path` starts with `prefix`.
pub fn has_prefix<P: AsRef<Path>>(path: P, prefix: &str) -> bool {
    path.as_ref().file_name().is_some_and(|name| name.to_str().unwrap().starts_with(prefix))
}

/// Returns true if the filename at `path` has the extension `extension`.
pub fn has_extension<P: AsRef<Path>>(path: P, extension: &str) -> bool {
    path.as_ref().extension().is_some_and(|ext| ext == extension)
}

/// Returns true if the filename at `path` ends with `suffix`.
pub fn has_suffix<P: AsRef<Path>>(path: P, suffix: &str) -> bool {
    path.as_ref().file_name().is_some_and(|name| name.to_str().unwrap().ends_with(suffix))
}

/// Returns true if the filename at `path` contains `needle`.
pub fn filename_contains<P: AsRef<Path>>(path: P, needle: &str) -> bool {
    path.as_ref().file_name().is_some_and(|name| name.to_str().unwrap().contains(needle))
}

/// Helper for reading entries in a given directory and its children.
pub fn read_dir_entries_recursive<P: AsRef<Path>, F: FnMut(&Path)>(dir: P, mut callback: F) {
    fn read_dir_entries_recursive_inner<P: AsRef<Path>, F: FnMut(&Path)>(dir: P, callback: &mut F) {
        for entry in rfs::read_dir(dir) {
            let path = entry.unwrap().path();
            callback(&path);
            if path.is_dir() {
                read_dir_entries_recursive_inner(path, callback);
            }
        }
    }

    read_dir_entries_recursive_inner(dir, &mut callback);
}
