pub use imp::env as imp;

pub mod traits {
    pub use super::Env as sys_Env;
}

pub mod prelude {
    pub use super::imp::{Env, SplitPaths, split_paths};
    pub use super::traits::*;
    pub use super::JoinPathsError;

    pub type Args = <Env as sys_Env>::Args;
    pub type Vars = <Env as sys_Env>::Vars;
}

use os_str::prelude::*;
use error::prelude::*;
use core::result;
use core::marker;
use core::fmt;
use core::iter::Iterator;
use core::convert::AsRef;

pub trait Env {
    type Args: Iterator<Item=OsString>;
    type Vars: Iterator<Item=(OsString, OsString)>;

    fn getcwd() -> Result<OsString>;
    fn chdir(p: &OsStr) -> Result<()>;

    fn getenv(k: &OsStr) -> Result<Option<OsString>>;
    fn setenv(k: &OsStr, v: &OsStr) -> Result<()>;
    fn unsetenv(k: &OsStr) -> Result<()>;

    fn home_dir() -> Result<OsString>;
    fn temp_dir() -> Result<OsString>;
    fn current_exe() -> Result<OsString>;

    fn env() -> Result<Self::Vars>;
    fn args() -> Result<Self::Args>;

    fn join_paths<'a, I: Iterator<Item=T>, T: AsRef<OsStr>>(paths: I) -> result::Result<OsString, JoinPathsError<Self>>;
    fn join_paths_error() -> &'static str;

    const ARCH: &'static str;
    const FAMILY: &'static str;
    const OS: &'static str;
    const DLL_PREFIX: &'static str;
    const DLL_SUFFIX: &'static str;
    const DLL_EXTENSION: &'static str;
    const EXE_SUFFIX: &'static str;
    const EXE_EXTENSION: &'static str;
}

pub struct JoinPathsError<E: Env + ?Sized>(marker::PhantomData<fn(&E)>);

impl<E: Env + ?Sized> fmt::Display for JoinPathsError<E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        E::join_paths_error().fmt(f)
    }
}

impl<E: Env + ?Sized> fmt::Debug for JoinPathsError<E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        E::join_paths_error().fmt(f)
    }
}

impl<E: Env + ?Sized> JoinPathsError<E> {
    pub const fn new() -> Self { JoinPathsError(marker::PhantomData) }
}

#[cfg(target_arch = "x86")]
mod arch {
    pub const ARCH: &'static str = "x86";
}

#[cfg(target_arch = "x86_64")]
mod arch {
    pub const ARCH: &'static str = "x86_64";
}

#[cfg(target_arch = "arm")]
mod arch {
    pub const ARCH: &'static str = "arm";
}

#[cfg(target_arch = "aarch64")]
mod arch {
    pub const ARCH: &'static str = "aarch64";
}

#[cfg(target_arch = "mips")]
mod arch {
    pub const ARCH: &'static str = "mips";
}

#[cfg(target_arch = "mipsel")]
mod arch {
    pub const ARCH: &'static str = "mipsel";
}

#[cfg(target_arch = "powerpc")]
mod arch {
    pub const ARCH: &'static str = "powerpc";
}

pub use self::arch::ARCH;
