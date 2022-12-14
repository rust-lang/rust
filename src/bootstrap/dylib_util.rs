// Various utilities for working with dylib paths.
//
// This file is meant to be included directly to avoid a dependency on the bootstrap library from
// the rustc and rustdoc wrappers. This improves compilation time by reducing the linking time.

/// Returns the environment variable which the dynamic library lookup path
/// resides in for this platform.
pub fn dylib_path_var() -> &'static str {
    if cfg!(target_os = "windows") {
        "PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else if cfg!(target_os = "haiku") {
        "LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}

/// Parses the `dylib_path_var()` environment variable, returning a list of
/// paths that are members of this lookup path.
pub fn dylib_path() -> Vec<PathBuf> {
    let var = match env::var_os(dylib_path_var()) {
        Some(v) => v,
        None => return vec![],
    };
    env::split_paths(&var).collect()
}
