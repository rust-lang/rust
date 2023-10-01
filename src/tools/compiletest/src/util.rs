use crate::common::Config;
use std::env;
use std::ffi::OsStr;
use std::path::PathBuf;
use std::process::Command;

use tracing::*;

#[cfg(test)]
mod tests;

pub fn make_new_path(path: &str) -> String {
    assert!(cfg!(windows));
    // Windows just uses PATH as the library search path, so we have to
    // maintain the current value while adding our own
    match env::var(lib_path_env_var()) {
        Ok(curr) => format!("{}{}{}", path, path_div(), curr),
        Err(..) => path.to_owned(),
    }
}

pub fn lib_path_env_var() -> &'static str {
    "PATH"
}
fn path_div() -> &'static str {
    ";"
}

pub fn logv(config: &Config, s: String) {
    debug!("{}", s);
    if config.verbose {
        println!("{}", s);
    }
}

pub trait PathBufExt {
    /// Append an extension to the path, even if it already has one.
    fn with_extra_extension<S: AsRef<OsStr>>(&self, extension: S) -> PathBuf;
}

impl PathBufExt for PathBuf {
    fn with_extra_extension<S: AsRef<OsStr>>(&self, extension: S) -> PathBuf {
        if extension.as_ref().is_empty() {
            self.clone()
        } else {
            let mut fname = self.file_name().unwrap().to_os_string();
            if !extension.as_ref().to_str().unwrap().starts_with('.') {
                fname.push(".");
            }
            fname.push(extension);
            self.with_file_name(fname)
        }
    }
}

/// The name of the environment variable that holds dynamic library locations.
pub fn dylib_env_var() -> &'static str {
    if cfg!(windows) {
        "PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else if cfg!(target_os = "haiku") {
        "LIBRARY_PATH"
    } else if cfg!(target_os = "aix") {
        "LIBPATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}

/// Adds a list of lookup paths to `cmd`'s dynamic library lookup path.
/// If the dylib_path_var is already set for this cmd, the old value will be overwritten!
pub fn add_dylib_path(cmd: &mut Command, paths: impl Iterator<Item = impl Into<PathBuf>>) {
    let path_env = env::var_os(dylib_env_var());
    let old_paths = path_env.as_ref().map(env::split_paths);
    let new_paths = paths.map(Into::into).chain(old_paths.into_iter().flatten());
    cmd.env(dylib_env_var(), env::join_paths(new_paths).unwrap());
}
