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
