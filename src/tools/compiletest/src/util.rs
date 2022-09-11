use crate::common::Config;
use std::env;
use std::ffi::OsStr;
use std::path::PathBuf;

use tracing::*;

#[cfg(test)]
mod tests;

pub const ASAN_SUPPORTED_TARGETS: &[&str] = &[
    "aarch64-apple-darwin",
    "aarch64-fuchsia",
    "aarch64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "x86_64-fuchsia",
    "x86_64-unknown-freebsd",
    "x86_64-unknown-linux-gnu",
];

// FIXME(rcvalle): More targets are likely supported.
pub const CFI_SUPPORTED_TARGETS: &[&str] = &[
    "aarch64-apple-darwin",
    "aarch64-fuchsia",
    "aarch64-linux-android",
    "aarch64-unknown-freebsd",
    "aarch64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "x86_64-fuchsia",
    "x86_64-pc-solaris",
    "x86_64-unknown-freebsd",
    "x86_64-unknown-illumos",
    "x86_64-unknown-linux-gnu",
    "x86_64-unknown-linux-musl",
    "x86_64-unknown-netbsd",
];

pub const LSAN_SUPPORTED_TARGETS: &[&str] = &[
    // FIXME: currently broken, see #88132
    // "aarch64-apple-darwin",
    "aarch64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-gnu",
];

pub const MSAN_SUPPORTED_TARGETS: &[&str] =
    &["aarch64-unknown-linux-gnu", "x86_64-unknown-freebsd", "x86_64-unknown-linux-gnu"];

pub const TSAN_SUPPORTED_TARGETS: &[&str] = &[
    "aarch64-apple-darwin",
    "aarch64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "x86_64-unknown-freebsd",
    "x86_64-unknown-linux-gnu",
];

pub const HWASAN_SUPPORTED_TARGETS: &[&str] =
    &["aarch64-linux-android", "aarch64-unknown-linux-gnu"];

pub const MEMTAG_SUPPORTED_TARGETS: &[&str] =
    &["aarch64-linux-android", "aarch64-unknown-linux-gnu"];

pub const SHADOWCALLSTACK_SUPPORTED_TARGETS: &[&str] = &["aarch64-linux-android"];

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
