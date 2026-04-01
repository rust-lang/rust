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

/// Check if staged `rustc`-under-test was built with debug assertions.
#[track_caller]
#[must_use]
pub fn rustc_debug_assertions_enabled() -> bool {
    // Note: we assume this env var is set when the test recipe is being executed.
    std::env::var_os("__RUSTC_DEBUG_ASSERTIONS_ENABLED").is_some()
}

/// Check if staged `std`-under-test was built with debug assertions.
#[track_caller]
#[must_use]
pub fn std_debug_assertions_enabled() -> bool {
    // Note: we assume this env var is set when the test recipe is being executed.
    std::env::var_os("__STD_DEBUG_ASSERTIONS_ENABLED").is_some()
}

/// Check if staged `std`-under-test was built with remapping of it's sources.
#[track_caller]
#[must_use]
pub fn std_remap_debuginfo_enabled() -> bool {
    // Note: we assume this env var is set when the test recipe is being executed.
    std::env::var_os("__STD_REMAP_DEBUGINFO_ENABLED").is_some()
}

/// A wrapper around [`std::env::set_current_dir`] which includes the directory
/// path in the panic message.
#[track_caller]
pub fn set_current_dir<P: AsRef<std::path::Path>>(dir: P) {
    std::env::set_current_dir(dir.as_ref())
        .expect(&format!("could not set current directory to \"{}\"", dir.as_ref().display()));
}

/// Number of parallel jobs bootstrap was configured with.
///
/// This may fallback to [`std::thread::available_parallelism`] when no explicit jobs count has been
/// configured. Refer to bootstrap's jobs fallback logic.
#[track_caller]
pub fn jobs() -> u32 {
    std::env::var_os("__BOOTSTRAP_JOBS")
        .expect("`__BOOTSTRAP_JOBS` must be set by `compiletest`")
        .to_str()
        .expect("`__BOOTSTRAP_JOBS` must be a valid string")
        .parse::<u32>()
        .expect("`__BOOTSTRAP_JOBS` must be a valid `u32`")
}
