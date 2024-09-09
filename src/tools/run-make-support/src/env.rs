use std::ffi::OsString;

#[track_caller]
#[must_use]
pub fn env_var(name: &str) -> String {
    match std::env::var(name) {
        Ok(v) => v,
        Err(err) => panic!("failed to retrieve environment variable {name:?}: {err:?}"),
    }
}

#[track_caller]
#[must_use]
pub fn env_var_os(name: &str) -> OsString {
    match std::env::var_os(name) {
        Some(v) => v,
        None => panic!("failed to retrieve environment variable {name:?}"),
    }
}

/// Check if `NO_DEBUG_ASSERTIONS` is set (usually this may be set in CI jobs).
#[track_caller]
#[must_use]
pub fn no_debug_assertions() -> bool {
    std::env::var_os("NO_DEBUG_ASSERTIONS").is_some()
}

/// A wrapper around [`std::env::set_current_dir`] which includes the directory
/// path in the panic message.
#[track_caller]
pub fn set_current_dir<P: AsRef<std::path::Path>>(dir: P) {
    std::env::set_current_dir(dir.as_ref())
        .expect(&format!("could not set current directory to \"{}\"", dir.as_ref().display()));
}
