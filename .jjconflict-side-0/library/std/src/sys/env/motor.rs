pub use super::common::Env;
use crate::ffi::{OsStr, OsString};
use crate::io;
use crate::os::motor::ffi::OsStrExt;

pub fn env() -> Env {
    let motor_env: Vec<(String, String)> = moto_rt::process::env();
    let mut rust_env = vec![];

    for (k, v) in motor_env {
        rust_env.push((OsString::from(k), OsString::from(v)));
    }

    Env::new(rust_env)
}

pub fn getenv(key: &OsStr) -> Option<OsString> {
    moto_rt::process::getenv(key.as_str()).map(|s| OsString::from(s))
}

pub unsafe fn setenv(key: &OsStr, val: &OsStr) -> io::Result<()> {
    Ok(moto_rt::process::setenv(key.as_str(), val.as_str()))
}

pub unsafe fn unsetenv(key: &OsStr) -> io::Result<()> {
    Ok(moto_rt::process::unsetenv(key.as_str()))
}
