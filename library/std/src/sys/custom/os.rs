#![allow(missing_docs)]

use crate::str::Split;
use crate::ffi::{OsStr, OsString};
use crate::path::{self, PathBuf};
use crate::custom_os_impl;
use crate::error::Error;
use crate::fmt;
use crate::vec;
use crate::io;

/// Inner content of [`crate::env::SplitPaths`]
pub struct SplitPaths<'a>(Split<'a, &'static str>);

impl<'a> Iterator for SplitPaths<'a> {
    type Item = PathBuf;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|s| s.into())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

/// Inner content of [`crate::env::JoinPathsError`]
#[derive(Debug)]
pub struct JoinPathsError(pub &'static str);

#[stable(feature = "env", since = "1.0.0")]
impl Error for JoinPathsError {
    #[allow(deprecated, deprecated_in_future)]
    fn description(&self) -> &str {
        self.0
    }
}

impl core::fmt::Display for JoinPathsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

pub fn errno() -> i32 {
    custom_os_impl!(os, errno)
}

pub fn error_string(errno: i32) -> String {
    custom_os_impl!(os, error_string, errno)
}

pub fn getcwd() -> io::Result<PathBuf> {
    custom_os_impl!(os, getcwd)
}

pub fn chdir(path: &path::Path) -> io::Result<()> {
    custom_os_impl!(os, chdir, path)
}

pub fn split_paths(unparsed: &OsStr) -> SplitPaths<'_> {
    let delim = custom_os_impl!(os, env_path_delim);
    let as_str = unparsed.to_str().expect("invalid PATH environment variable");

    SplitPaths(as_str.split(delim))
}

pub fn join_paths<I, T>(mut paths: I) -> Result<OsString, JoinPathsError>
where
    I: Iterator<Item = T>,
    T: AsRef<OsStr>,
{
    let delim = custom_os_impl!(os, env_path_delim);
    let mut joined = match paths.next() {
        Some(first) => first.as_ref().to_os_string(),
        None => OsString::new(),
    };

    while let Some(path) = paths.next() {
        joined.push(delim);
        joined.push(path);
    }

    Ok(joined)
}

pub fn current_exe() -> io::Result<PathBuf> {
    custom_os_impl!(os, current_exe)
}

/// An environment variable definition
#[derive(Debug)]
pub struct Variable {
    pub name: OsString,
    pub value: OsString,
}

/// An Iterator on environment variables
pub struct Env {
    pub iter: vec::IntoIter<Variable>,
}

// FIXME(https://github.com/rust-lang/rust/issues/114583): Remove this when <OsStr as Debug>::fmt matches <str as Debug>::fmt.
pub struct EnvStrDebug<'a> {
    slice: &'a [Variable],
}

impl fmt::Debug for EnvStrDebug<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { slice } = self;
        f.debug_list()
            .entries(slice.iter().map(|var| (var.name.to_str().unwrap(), var.value.to_str().unwrap())))
            .finish()
    }
}

impl Env {
    pub fn str_debug(&self) -> impl fmt::Debug + '_ {
        let Self { iter } = self;
        EnvStrDebug { slice: iter.as_slice() }
    }
}

impl fmt::Debug for Env {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { iter } = self;
        f.debug_list().entries(iter.as_slice()).finish()
    }
}

impl !Send for Env {}
impl !Sync for Env {}

impl Iterator for Env {
    type Item = (OsString, OsString);
    fn next(&mut self) -> Option<(OsString, OsString)> {
        self.iter.next().map(|var| (var.name, var.value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

pub fn env() -> Env {
    custom_os_impl!(os, env)
}

pub fn getenv(var_name: &OsStr) -> Option<OsString> {
    custom_os_impl!(os, get_env, var_name)
}

pub fn setenv(var_name: &OsStr, value: &OsStr) -> io::Result<()> {
    custom_os_impl!(os, set_env, var_name, value)
}

pub fn unsetenv(var_name: &OsStr) -> io::Result<()> {
    custom_os_impl!(os, unset_env, var_name)
}

pub fn temp_dir() -> PathBuf {
    custom_os_impl!(os, temp_dir)
}

pub fn home_dir() -> Option<PathBuf> {
    custom_os_impl!(os, home_dir)
}

pub fn exit(code: i32) -> ! {
    custom_os_impl!(os, exit, code)
}

pub fn getpid() -> u32 {
    custom_os_impl!(os, get_pid)
}
