use std::ffi::OsString;
use std::env;

#[track_caller]
#[must_use]
pub fn env_var(name: &str) -> String {
    match env::var(name) {
        Ok(v) => v,
        Err(err) => panic!("failed to retrieve environment variable {name:?}: {err:?}"),
    }
}

#[track_caller]
#[must_use]
pub fn env_var_os(name: &str) -> OsString {
    match env::var_os(name) {
        Some(v) => v,
        None => panic!("failed to retrieve environment variable {name:?}"),
    }
}
