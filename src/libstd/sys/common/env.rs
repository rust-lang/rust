use fmt;
use sys::env;

pub struct JoinPathsError(());

impl fmt::Display for JoinPathsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        env::join_paths_error().fmt(f)
    }
}

impl fmt::Debug for JoinPathsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        env::join_paths_error().fmt(f)
    }
}

impl JoinPathsError {
    pub const fn new() -> Self { JoinPathsError(()) }
}

#[cfg(target_arch = "x86")]
pub const ARCH: &'static str = "x86";

#[cfg(target_arch = "x86_64")]
pub const ARCH: &'static str = "x86_64";

#[cfg(target_arch = "arm")]
pub const ARCH: &'static str = "arm";

#[cfg(target_arch = "aarch64")]
pub const ARCH: &'static str = "aarch64";

#[cfg(target_arch = "mips")]
pub const ARCH: &'static str = "mips";

#[cfg(target_arch = "mipsel")]
pub const ARCH: &'static str = "mipsel";

#[cfg(target_arch = "powerpc")]
pub const ARCH: &'static str = "powerpc";
