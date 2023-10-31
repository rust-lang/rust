use crate::common::Config;
use std::env;
use std::ffi::OsStr;
use std::path::PathBuf;
use std::process::Command;

use tracing::*;

#[cfg(test)]
mod tests;

pub const ASAN_SUPPORTED_TARGETS: &[&str] = &[
    "aarch64-apple-darwin",
    "aarch64-apple-ios",
    "aarch64-apple-ios-sim",
    "aarch64-unknown-fuchsia",
    "aarch64-linux-android",
    "aarch64-unknown-linux-gnu",
    "arm-linux-androideabi",
    "armv7-linux-androideabi",
    "i686-linux-android",
    "i686-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "x86_64-apple-ios",
    "x86_64-unknown-fuchsia",
    "x86_64-linux-android",
    "x86_64-unknown-freebsd",
    "x86_64-unknown-linux-gnu",
    "s390x-unknown-linux-gnu",
];

// FIXME(rcvalle): More targets are likely supported.
pub const CFI_SUPPORTED_TARGETS: &[&str] = &[
    "aarch64-apple-darwin",
    "aarch64-unknown-fuchsia",
    "aarch64-linux-android",
    "aarch64-unknown-freebsd",
    "aarch64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "x86_64-unknown-fuchsia",
    "x86_64-pc-solaris",
    "x86_64-unknown-freebsd",
    "x86_64-unknown-illumos",
    "x86_64-unknown-linux-gnu",
    "x86_64-unknown-linux-musl",
    "x86_64-unknown-netbsd",
];

pub const KCFI_SUPPORTED_TARGETS: &[&str] = &["aarch64-linux-none", "x86_64-linux-none"];

pub const KASAN_SUPPORTED_TARGETS: &[&str] = &[
    "aarch64-unknown-none",
    "riscv64gc-unknown-none-elf",
    "riscv64imac-unknown-none-elf",
    "x86_64-unknown-none",
];

pub const LSAN_SUPPORTED_TARGETS: &[&str] = &[
    // FIXME: currently broken, see #88132
    // "aarch64-apple-darwin",
    "aarch64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-gnu",
    "s390x-unknown-linux-gnu",
];

pub const MSAN_SUPPORTED_TARGETS: &[&str] = &[
    "aarch64-unknown-linux-gnu",
    "x86_64-unknown-freebsd",
    "x86_64-unknown-linux-gnu",
    "s390x-unknown-linux-gnu",
];

pub const TSAN_SUPPORTED_TARGETS: &[&str] = &[
    "aarch64-apple-darwin",
    "aarch64-apple-ios",
    "aarch64-apple-ios-sim",
    "aarch64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "x86_64-apple-ios",
    "x86_64-unknown-freebsd",
    "x86_64-unknown-linux-gnu",
    "s390x-unknown-linux-gnu",
];

pub const HWASAN_SUPPORTED_TARGETS: &[&str] =
    &["aarch64-linux-android", "aarch64-unknown-linux-gnu"];

pub const MEMTAG_SUPPORTED_TARGETS: &[&str] =
    &["aarch64-linux-android", "aarch64-unknown-linux-gnu"];

pub const SHADOWCALLSTACK_SUPPORTED_TARGETS: &[&str] = &["aarch64-linux-android"];

pub const XRAY_SUPPORTED_TARGETS: &[&str] = &[
    "aarch64-linux-android",
    "aarch64-unknown-linux-gnu",
    "aarch64-unknown-linux-musl",
    "x86_64-linux-android",
    "x86_64-unknown-freebsd",
    "x86_64-unknown-linux-gnu",
    "x86_64-unknown-linux-musl",
    "x86_64-unknown-netbsd",
    "x86_64-unknown-none-linuxkernel",
    "x86_64-unknown-openbsd",
];

pub const SAFESTACK_SUPPORTED_TARGETS: &[&str] = &["x86_64-unknown-linux-gnu"];

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
