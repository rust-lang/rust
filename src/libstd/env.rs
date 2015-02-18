// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Inspection and manipulation of the process's environment.
//!
//! This module contains methods to inspect various aspects such as
//! environment varibles, process arguments, the current directory, and various
//! other important directories.

#![unstable(feature = "env", reason = "recently added via RFC 578")]

use prelude::v1::*;

use error::Error;
use ffi::{OsString, AsOsStr};
use fmt;
use old_io::IoResult;
use sync::atomic::{AtomicIsize, ATOMIC_ISIZE_INIT, Ordering};
use sync::{StaticMutex, MUTEX_INIT};
use sys::os as os_imp;

/// Returns the current working directory as a `Path`.
///
/// # Errors
///
/// Returns an `Err` if the current working directory value is invalid.
/// Possible cases:
///
/// * Current directory does not exist.
/// * There are insufficient permissions to access the current directory.
/// * The internal buffer is not large enough to hold the path.
///
/// # Example
///
/// ```rust
/// use std::env;
///
/// // We assume that we are in a valid directory.
/// let p = env::current_dir().unwrap();
/// println!("The current directory is {}", p.display());
/// ```
pub fn current_dir() -> IoResult<Path> {
    os_imp::getcwd()
}

/// Changes the current working directory to the specified path, returning
/// whether the change was completed successfully or not.
///
/// # Example
///
/// ```rust
/// use std::env;
/// use std::old_path::Path;
///
/// let root = Path::new("/");
/// assert!(env::set_current_dir(&root).is_ok());
/// println!("Successfully changed working directory to {}!", root.display());
/// ```
pub fn set_current_dir(p: &Path) -> IoResult<()> {
    os_imp::chdir(p)
}

static ENV_LOCK: StaticMutex = MUTEX_INIT;

/// An iterator over a snapshot of the environment variables of this process.
///
/// This iterator is created through `std::env::vars()` and yields `(String,
/// String)` pairs.
pub struct Vars { inner: VarsOs }

/// An iterator over a snapshot of the environment variables of this process.
///
/// This iterator is created through `std::env::vars_os()` and yields
/// `(OsString, OsString)` pairs.
pub struct VarsOs { inner: os_imp::Env }

/// Returns an iterator of (variable, value) pairs of strings, for all the
/// environment variables of the current process.
///
/// The returned iterator contains a snapshot of the process's environment
/// variables at the time of this invocation, modifications to environment
/// variables afterwards will not be reflected in the returned iterator.
///
/// # Panics
///
/// While iterating, the returned iterator will panic if any key or value in the
/// environment is not valid unicode. If this is not desired, consider using the
/// `env::vars_os` function.
///
/// # Example
///
/// ```rust
/// use std::env;
///
/// // We will iterate through the references to the element returned by
/// // env::vars();
/// for (key, value) in env::vars() {
///     println!("{}: {}", key, value);
/// }
/// ```
pub fn vars() -> Vars {
    Vars { inner: vars_os() }
}

/// Returns an iterator of (variable, value) pairs of OS strings, for all the
/// environment variables of the current process.
///
/// The returned iterator contains a snapshot of the process's environment
/// variables at the time of this invocation, modifications to environment
/// variables afterwards will not be reflected in the returned iterator.
///
/// # Example
///
/// ```rust
/// use std::env;
///
/// // We will iterate through the references to the element returned by
/// // env::vars_os();
/// for (key, value) in env::vars_os() {
///     println!("{:?}: {:?}", key, value);
/// }
/// ```
pub fn vars_os() -> VarsOs {
    let _g = ENV_LOCK.lock();
    VarsOs { inner: os_imp::env() }
}

impl Iterator for Vars {
    type Item = (String, String);
    fn next(&mut self) -> Option<(String, String)> {
        self.inner.next().map(|(a, b)| {
            (a.into_string().unwrap(), b.into_string().unwrap())
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}

impl Iterator for VarsOs {
    type Item = (OsString, OsString);
    fn next(&mut self) -> Option<(OsString, OsString)> { self.inner.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}

/// Fetches the environment variable `key` from the current process.
///
/// The returned result is `Ok(s)` if the environment variable is present and is
/// valid unicode. If the environment variable is not present, or it is not
/// valid unicode, then `Err` will be returned.
///
/// # Example
///
/// ```rust
/// use std::env;
///
/// let key = "HOME";
/// match env::var(key) {
///     Ok(val) => println!("{}: {:?}", key, val),
///     Err(e) => println!("couldn't interpret {}: {}", key, e),
/// }
/// ```
pub fn var<K: ?Sized>(key: &K) -> Result<String, VarError> where K: AsOsStr {
    match var_os(key) {
        Some(s) => s.into_string().map_err(VarError::NotUnicode),
        None => Err(VarError::NotPresent)
    }
}

/// Fetches the environment variable `key` from the current process, returning
/// None if the variable isn't set.
///
/// # Example
///
/// ```rust
/// use std::env;
///
/// let key = "HOME";
/// match env::var_os(key) {
///     Some(val) => println!("{}: {:?}", key, val),
///     None => println!("{} is not defined in the environment.", key)
/// }
/// ```
pub fn var_os<K: ?Sized>(key: &K) -> Option<OsString> where K: AsOsStr {
    let _g = ENV_LOCK.lock();
    os_imp::getenv(key.as_os_str())
}

/// Possible errors from the `env::var` method.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum VarError {
    /// The specified environment variable was not present in the current
    /// process's environment.
    NotPresent,

    /// The specified environment variable was found, but it did not contain
    /// valid unicode data. The found data is returned as a payload of this
    /// variant.
    NotUnicode(OsString),
}

impl fmt::Display for VarError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            VarError::NotPresent => write!(f, "environment variable not found"),
            VarError::NotUnicode(ref s) => {
                write!(f, "environment variable was not valid unicode: {:?}", s)
            }
        }
    }
}

impl Error for VarError {
    fn description(&self) -> &str {
        match *self {
            VarError::NotPresent => "environment variable not found",
            VarError::NotUnicode(..) => "environment variable was not valid unicode",
        }
    }
}

/// Sets the environment variable `k` to the value `v` for the currently running
/// process.
///
/// # Example
///
/// ```rust
/// use std::env;
///
/// let key = "KEY";
/// env::set_var(key, "VALUE");
/// assert_eq!(env::var(key), Ok("VALUE".to_string()));
/// ```
pub fn set_var<K: ?Sized, V: ?Sized>(k: &K, v: &V)
    where K: AsOsStr, V: AsOsStr
{
    let _g = ENV_LOCK.lock();
    os_imp::setenv(k.as_os_str(), v.as_os_str())
}

/// Remove a variable from the environment entirely.
pub fn remove_var<K: ?Sized>(k: &K) where K: AsOsStr {
    let _g = ENV_LOCK.lock();
    os_imp::unsetenv(k.as_os_str())
}

/// An iterator over `Path` instances for parsing an environment variable
/// according to platform-specific conventions.
///
/// This structure is returned from `std::env::split_paths`.
pub struct SplitPaths<'a> { inner: os_imp::SplitPaths<'a> }

/// Parses input according to platform conventions for the `PATH`
/// environment variable.
///
/// Returns an iterator over the paths contained in `unparsed`.
///
/// # Example
///
/// ```rust
/// use std::env;
///
/// let key = "PATH";
/// match env::var_os(key) {
///     Some(paths) => {
///         for path in env::split_paths(&paths) {
///             println!("'{}'", path.display());
///         }
///     }
///     None => println!("{} is not defined in the environment.", key)
/// }
/// ```
pub fn split_paths<T: AsOsStr + ?Sized>(unparsed: &T) -> SplitPaths {
    SplitPaths { inner: os_imp::split_paths(unparsed.as_os_str()) }
}

impl<'a> Iterator for SplitPaths<'a> {
    type Item = Path;
    fn next(&mut self) -> Option<Path> { self.inner.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}

/// Error type returned from `std::env::join_paths` when paths fail to be
/// joined.
#[derive(Debug)]
pub struct JoinPathsError {
    inner: os_imp::JoinPathsError
}

/// Joins a collection of `Path`s appropriately for the `PATH`
/// environment variable.
///
/// Returns an `OsString` on success.
///
/// Returns an `Err` (containing an error message) if one of the input
/// `Path`s contains an invalid character for constructing the `PATH`
/// variable (a double quote on Windows or a colon on Unix).
///
/// # Example
///
/// ```rust
/// use std::env;
///
/// if let Some(path) = env::var_os("PATH") {
///     let mut paths = env::split_paths(&path).collect::<Vec<_>>();
///     paths.push(Path::new("/home/xyz/bin"));
///     let new_path = env::join_paths(paths.iter()).unwrap();
///     env::set_var("PATH", &new_path);
/// }
/// ```
pub fn join_paths<I, T>(paths: I) -> Result<OsString, JoinPathsError>
    where I: Iterator<Item=T>, T: AsOsStr
{
    os_imp::join_paths(paths).map_err(|e| {
        JoinPathsError { inner: e }
    })
}

impl fmt::Display for JoinPathsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.inner.fmt(f)
    }
}

impl Error for JoinPathsError {
    fn description(&self) -> &str { self.inner.description() }
}

/// Optionally returns the path to the current user's home directory if known.
///
/// # Unix
///
/// Returns the value of the 'HOME' environment variable if it is set
/// and not equal to the empty string.
///
/// # Windows
///
/// Returns the value of the 'HOME' environment variable if it is
/// set and not equal to the empty string. Otherwise, returns the value of the
/// 'USERPROFILE' environment variable if it is set and not equal to the empty
/// string.
///
/// # Example
///
/// ```rust
/// use std::env;
///
/// match env::home_dir() {
///     Some(ref p) => println!("{}", p.display()),
///     None => println!("Impossible to get your home dir!")
/// }
/// ```
pub fn home_dir() -> Option<Path> {
    os_imp::home_dir()
}

/// Returns the path to a temporary directory.
///
/// On Unix, returns the value of the 'TMPDIR' environment variable if it is
/// set, otherwise for non-Android it returns '/tmp'. If Android, since there
/// is no global temporary folder (it is usually allocated per-app), we return
/// '/data/local/tmp'.
///
/// On Windows, returns the value of, in order, the 'TMP', 'TEMP',
/// 'USERPROFILE' environment variable  if any are set and not the empty
/// string. Otherwise, tmpdir returns the path to the Windows directory.
pub fn temp_dir() -> Path {
    os_imp::temp_dir()
}

/// Optionally returns the filesystem path to the current executable which is
/// running but with the executable name.
///
/// The path returned is not necessarily a "real path" to the executable as
/// there may be intermediate symlinks.
///
/// # Errors
///
/// Acquiring the path to the current executable is a platform-specific operation
/// that can fail for a good number of reasons. Some errors can include, but not
/// be limited to filesystem operations failing or general syscall failures.
///
/// # Examples
///
/// ```rust
/// use std::env;
///
/// match env::current_exe() {
///     Ok(exe_path) => println!("Path of this executable is: {}",
///                               exe_path.display()),
///     Err(e) => println!("failed to get current exe path: {}", e),
/// };
/// ```
pub fn current_exe() -> IoResult<Path> {
    os_imp::current_exe()
}

static EXIT_STATUS: AtomicIsize = ATOMIC_ISIZE_INIT;

/// Sets the process exit code
///
/// Sets the exit code returned by the process if all supervised tasks
/// terminate successfully (without panicking). If the current root task panics
/// and is supervised by the scheduler then any user-specified exit status is
/// ignored and the process exits with the default panic status.
///
/// Note that this is not synchronized against modifications of other threads.
pub fn set_exit_status(code: i32) {
    EXIT_STATUS.store(code as isize, Ordering::SeqCst)
}

/// Fetches the process's current exit code. This defaults to 0 and can change
/// by calling `set_exit_status`.
pub fn get_exit_status() -> i32 {
    EXIT_STATUS.load(Ordering::SeqCst) as i32
}

/// An iterator over the arguments of a process, yielding an `String` value
/// for each argument.
///
/// This structure is created through the `std::env::args` method.
pub struct Args { inner: ArgsOs }

/// An iterator over the arguments of a process, yielding an `OsString` value
/// for each argument.
///
/// This structure is created through the `std::env::args_os` method.
pub struct ArgsOs { inner: os_imp::Args }

/// Returns the arguments which this program was started with (normally passed
/// via the command line).
///
/// The first element is traditionally the path to the executable, but it can be
/// set to arbitrary text, and it may not even exist, so this property should
/// not be relied upon for security purposes.
///
/// # Panics
///
/// The returned iterator will panic during iteration if any argument to the
/// process is not valid unicode. If this is not desired it is recommended to
/// use the `args_os` function instead.
///
/// # Example
///
/// ```rust
/// use std::env;
///
/// // Prints each argument on a separate line
/// for argument in env::args() {
///     println!("{}", argument);
/// }
/// ```
pub fn args() -> Args {
    Args { inner: args_os() }
}

/// Returns the arguments which this program was started with (normally passed
/// via the command line).
///
/// The first element is traditionally the path to the executable, but it can be
/// set to arbitrary text, and it may not even exist, so this property should
/// not be relied upon for security purposes.
///
/// # Example
///
/// ```rust
/// use std::env;
///
/// // Prints each argument on a separate line
/// for argument in env::args_os() {
///     println!("{:?}", argument);
/// }
/// ```
pub fn args_os() -> ArgsOs {
    ArgsOs { inner: os_imp::args() }
}

impl Iterator for Args {
    type Item = String;
    fn next(&mut self) -> Option<String> {
        self.inner.next().map(|s| s.into_string().unwrap())
    }
    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}

impl ExactSizeIterator for Args {
    fn len(&self) -> usize { self.inner.len() }
}

impl Iterator for ArgsOs {
    type Item = OsString;
    fn next(&mut self) -> Option<OsString> { self.inner.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}

impl ExactSizeIterator for ArgsOs {
    fn len(&self) -> usize { self.inner.len() }
}

/// Returns the page size of the current architecture in bytes.
pub fn page_size() -> usize {
    os_imp::page_size()
}

/// Constants associated with the current target
#[cfg(target_os = "linux")]
pub mod consts {
    pub use super::arch_consts::ARCH;

    pub const FAMILY: &'static str = "unix";

    /// A string describing the specific operating system in use: in this
    /// case, `linux`.
    pub const OS: &'static str = "linux";

    /// Specifies the filename prefix used for shared libraries on this
    /// platform: in this case, `lib`.
    pub const DLL_PREFIX: &'static str = "lib";

    /// Specifies the filename suffix used for shared libraries on this
    /// platform: in this case, `.so`.
    pub const DLL_SUFFIX: &'static str = ".so";

    /// Specifies the file extension used for shared libraries on this
    /// platform that goes after the dot: in this case, `so`.
    pub const DLL_EXTENSION: &'static str = "so";

    /// Specifies the filename suffix used for executable binaries on this
    /// platform: in this case, the empty string.
    pub const EXE_SUFFIX: &'static str = "";

    /// Specifies the file extension, if any, used for executable binaries
    /// on this platform: in this case, the empty string.
    pub const EXE_EXTENSION: &'static str = "";
}

/// Constants associated with the current target
#[cfg(target_os = "macos")]
pub mod consts {
    pub use super::arch_consts::ARCH;

    pub const FAMILY: &'static str = "unix";

    /// A string describing the specific operating system in use: in this
    /// case, `macos`.
    pub const OS: &'static str = "macos";

    /// Specifies the filename prefix used for shared libraries on this
    /// platform: in this case, `lib`.
    pub const DLL_PREFIX: &'static str = "lib";

    /// Specifies the filename suffix used for shared libraries on this
    /// platform: in this case, `.dylib`.
    pub const DLL_SUFFIX: &'static str = ".dylib";

    /// Specifies the file extension used for shared libraries on this
    /// platform that goes after the dot: in this case, `dylib`.
    pub const DLL_EXTENSION: &'static str = "dylib";

    /// Specifies the filename suffix used for executable binaries on this
    /// platform: in this case, the empty string.
    pub const EXE_SUFFIX: &'static str = "";

    /// Specifies the file extension, if any, used for executable binaries
    /// on this platform: in this case, the empty string.
    pub const EXE_EXTENSION: &'static str = "";
}

/// Constants associated with the current target
#[cfg(target_os = "ios")]
pub mod consts {
    pub use super::arch_consts::ARCH;

    pub const FAMILY: &'static str = "unix";

    /// A string describing the specific operating system in use: in this
    /// case, `ios`.
    pub const OS: &'static str = "ios";

    /// Specifies the filename suffix used for executable binaries on this
    /// platform: in this case, the empty string.
    pub const EXE_SUFFIX: &'static str = "";

    /// Specifies the file extension, if any, used for executable binaries
    /// on this platform: in this case, the empty string.
    pub const EXE_EXTENSION: &'static str = "";
}

/// Constants associated with the current target
#[cfg(target_os = "freebsd")]
pub mod consts {
    pub use super::arch_consts::ARCH;

    pub const FAMILY: &'static str = "unix";

    /// A string describing the specific operating system in use: in this
    /// case, `freebsd`.
    pub const OS: &'static str = "freebsd";

    /// Specifies the filename prefix used for shared libraries on this
    /// platform: in this case, `lib`.
    pub const DLL_PREFIX: &'static str = "lib";

    /// Specifies the filename suffix used for shared libraries on this
    /// platform: in this case, `.so`.
    pub const DLL_SUFFIX: &'static str = ".so";

    /// Specifies the file extension used for shared libraries on this
    /// platform that goes after the dot: in this case, `so`.
    pub const DLL_EXTENSION: &'static str = "so";

    /// Specifies the filename suffix used for executable binaries on this
    /// platform: in this case, the empty string.
    pub const EXE_SUFFIX: &'static str = "";

    /// Specifies the file extension, if any, used for executable binaries
    /// on this platform: in this case, the empty string.
    pub const EXE_EXTENSION: &'static str = "";
}

/// Constants associated with the current target
#[cfg(target_os = "dragonfly")]
pub mod consts {
    pub use super::arch_consts::ARCH;

    pub const FAMILY: &'static str = "unix";

    /// A string describing the specific operating system in use: in this
    /// case, `dragonfly`.
    pub const OS: &'static str = "dragonfly";

    /// Specifies the filename prefix used for shared libraries on this
    /// platform: in this case, `lib`.
    pub const DLL_PREFIX: &'static str = "lib";

    /// Specifies the filename suffix used for shared libraries on this
    /// platform: in this case, `.so`.
    pub const DLL_SUFFIX: &'static str = ".so";

    /// Specifies the file extension used for shared libraries on this
    /// platform that goes after the dot: in this case, `so`.
    pub const DLL_EXTENSION: &'static str = "so";

    /// Specifies the filename suffix used for executable binaries on this
    /// platform: in this case, the empty string.
    pub const EXE_SUFFIX: &'static str = "";

    /// Specifies the file extension, if any, used for executable binaries
    /// on this platform: in this case, the empty string.
    pub const EXE_EXTENSION: &'static str = "";
}

/// Constants associated with the current target
#[cfg(target_os = "openbsd")]
pub mod consts {
    pub use super::arch_consts::ARCH;

    pub const FAMILY: &'static str = "unix";

    /// A string describing the specific operating system in use: in this
    /// case, `dragonfly`.
    pub const OS: &'static str = "openbsd";

    /// Specifies the filename prefix used for shared libraries on this
    /// platform: in this case, `lib`.
    pub const DLL_PREFIX: &'static str = "lib";

    /// Specifies the filename suffix used for shared libraries on this
    /// platform: in this case, `.so`.
    pub const DLL_SUFFIX: &'static str = ".so";

    /// Specifies the file extension used for shared libraries on this
    /// platform that goes after the dot: in this case, `so`.
    pub const DLL_EXTENSION: &'static str = "so";

    /// Specifies the filename suffix used for executable binaries on this
    /// platform: in this case, the empty string.
    pub const EXE_SUFFIX: &'static str = "";

    /// Specifies the file extension, if any, used for executable binaries
    /// on this platform: in this case, the empty string.
    pub const EXE_EXTENSION: &'static str = "";
}

/// Constants associated with the current target
#[cfg(target_os = "android")]
pub mod consts {
    pub use super::arch_consts::ARCH;

    pub const FAMILY: &'static str = "unix";

    /// A string describing the specific operating system in use: in this
    /// case, `android`.
    pub const OS: &'static str = "android";

    /// Specifies the filename prefix used for shared libraries on this
    /// platform: in this case, `lib`.
    pub const DLL_PREFIX: &'static str = "lib";

    /// Specifies the filename suffix used for shared libraries on this
    /// platform: in this case, `.so`.
    pub const DLL_SUFFIX: &'static str = ".so";

    /// Specifies the file extension used for shared libraries on this
    /// platform that goes after the dot: in this case, `so`.
    pub const DLL_EXTENSION: &'static str = "so";

    /// Specifies the filename suffix used for executable binaries on this
    /// platform: in this case, the empty string.
    pub const EXE_SUFFIX: &'static str = "";

    /// Specifies the file extension, if any, used for executable binaries
    /// on this platform: in this case, the empty string.
    pub const EXE_EXTENSION: &'static str = "";
}

/// Constants associated with the current target
#[cfg(target_os = "windows")]
pub mod consts {
    pub use super::arch_consts::ARCH;

    pub const FAMILY: &'static str = "windows";

    /// A string describing the specific operating system in use: in this
    /// case, `windows`.
    pub const OS: &'static str = "windows";

    /// Specifies the filename prefix used for shared libraries on this
    /// platform: in this case, the empty string.
    pub const DLL_PREFIX: &'static str = "";

    /// Specifies the filename suffix used for shared libraries on this
    /// platform: in this case, `.dll`.
    pub const DLL_SUFFIX: &'static str = ".dll";

    /// Specifies the file extension used for shared libraries on this
    /// platform that goes after the dot: in this case, `dll`.
    pub const DLL_EXTENSION: &'static str = "dll";

    /// Specifies the filename suffix used for executable binaries on this
    /// platform: in this case, `.exe`.
    pub const EXE_SUFFIX: &'static str = ".exe";

    /// Specifies the file extension, if any, used for executable binaries
    /// on this platform: in this case, `exe`.
    pub const EXE_EXTENSION: &'static str = "exe";
}

#[cfg(target_arch = "x86")]
mod arch_consts {
    pub const ARCH: &'static str = "x86";
}

#[cfg(target_arch = "x86_64")]
mod arch_consts {
    pub const ARCH: &'static str = "x86_64";
}

#[cfg(target_arch = "arm")]
mod arch_consts {
    pub const ARCH: &'static str = "arm";
}

#[cfg(target_arch = "aarch64")]
mod arch_consts {
    pub const ARCH: &'static str = "aarch64";
}

#[cfg(target_arch = "mips")]
mod arch_consts {
    pub const ARCH: &'static str = "mips";
}

#[cfg(target_arch = "mipsel")]
mod arch_consts {
    pub const ARCH: &'static str = "mipsel";
}

#[cfg(target_arch = "powerpc")]
mod arch_consts {
    pub const ARCH: &'static str = "powerpc";
}

#[cfg(test)]
mod tests {
    use prelude::v1::*;
    use super::*;
    use iter::repeat;
    use rand::{self, Rng};
    use ffi::{OsString, OsStr};

    fn make_rand_name() -> OsString {
        let mut rng = rand::thread_rng();
        let n = format!("TEST{}", rng.gen_ascii_chars().take(10)
                                     .collect::<String>());
        let n = OsString::from_string(n);
        assert!(var_os(&n).is_none());
        n
    }

    fn eq(a: Option<OsString>, b: Option<&str>) {
        assert_eq!(a.as_ref().map(|s| &**s), b.map(OsStr::from_str).map(|s| &*s));
    }

    #[test]
    fn test_set_var() {
        let n = make_rand_name();
        set_var(&n, "VALUE");
        eq(var_os(&n), Some("VALUE"));
    }

    #[test]
    fn test_remove_var() {
        let n = make_rand_name();
        set_var(&n, "VALUE");
        remove_var(&n);
        eq(var_os(&n), None);
    }

    #[test]
    fn test_set_var_overwrite() {
        let n = make_rand_name();
        set_var(&n, "1");
        set_var(&n, "2");
        eq(var_os(&n), Some("2"));
        set_var(&n, "");
        eq(var_os(&n), Some(""));
    }

    #[test]
    fn test_var_big() {
        let mut s = "".to_string();
        let mut i = 0;
        while i < 100 {
            s.push_str("aaaaaaaaaa");
            i += 1;
        }
        let n = make_rand_name();
        set_var(&n, s.as_slice());
        eq(var_os(&n), Some(s.as_slice()));
    }

    #[test]
    fn test_self_exe_path() {
        let path = current_exe();
        assert!(path.is_ok());
        let path = path.unwrap();

        // Hard to test this function
        assert!(path.is_absolute());
    }

    #[test]
    fn test_env_set_get_huge() {
        let n = make_rand_name();
        let s = repeat("x").take(10000).collect::<String>();
        set_var(&n, &s);
        eq(var_os(&n), Some(s.as_slice()));
        remove_var(&n);
        eq(var_os(&n), None);
    }

    #[test]
    fn test_env_set_var() {
        let n = make_rand_name();

        let mut e = vars_os();
        set_var(&n, "VALUE");
        assert!(!e.any(|(k, v)| {
            &*k == &*n && &*v == "VALUE"
        }));

        assert!(vars_os().any(|(k, v)| {
            &*k == &*n && &*v == "VALUE"
        }));
    }

    #[test]
    fn test() {
        assert!((!Path::new("test-path").is_absolute()));

        current_dir().unwrap();
    }

    #[test]
    #[cfg(windows)]
    fn split_paths_windows() {
        fn check_parse(unparsed: &str, parsed: &[&str]) -> bool {
            split_paths(unparsed).collect::<Vec<_>>() ==
                parsed.iter().map(|s| Path::new(*s)).collect::<Vec<_>>()
        }

        assert!(check_parse("", &mut [""]));
        assert!(check_parse(r#""""#, &mut [""]));
        assert!(check_parse(";;", &mut ["", "", ""]));
        assert!(check_parse(r"c:\", &mut [r"c:\"]));
        assert!(check_parse(r"c:\;", &mut [r"c:\", ""]));
        assert!(check_parse(r"c:\;c:\Program Files\",
                            &mut [r"c:\", r"c:\Program Files\"]));
        assert!(check_parse(r#"c:\;c:\"foo"\"#, &mut [r"c:\", r"c:\foo\"]));
        assert!(check_parse(r#"c:\;c:\"foo;bar"\;c:\baz"#,
                            &mut [r"c:\", r"c:\foo;bar\", r"c:\baz"]));
    }

    #[test]
    #[cfg(unix)]
    fn split_paths_unix() {
        fn check_parse(unparsed: &str, parsed: &[&str]) -> bool {
            split_paths(unparsed).collect::<Vec<_>>() ==
                parsed.iter().map(|s| Path::new(*s)).collect::<Vec<_>>()
        }

        assert!(check_parse("", &mut [""]));
        assert!(check_parse("::", &mut ["", "", ""]));
        assert!(check_parse("/", &mut ["/"]));
        assert!(check_parse("/:", &mut ["/", ""]));
        assert!(check_parse("/:/usr/local", &mut ["/", "/usr/local"]));
    }

    #[test]
    #[cfg(unix)]
    fn join_paths_unix() {
        fn test_eq(input: &[&str], output: &str) -> bool {
            &*join_paths(input.iter().cloned()).unwrap() ==
                OsStr::from_str(output)
        }

        assert!(test_eq(&[], ""));
        assert!(test_eq(&["/bin", "/usr/bin", "/usr/local/bin"],
                         "/bin:/usr/bin:/usr/local/bin"));
        assert!(test_eq(&["", "/bin", "", "", "/usr/bin", ""],
                         ":/bin:::/usr/bin:"));
        assert!(join_paths(["/te:st"].iter().cloned()).is_err());
    }

    #[test]
    #[cfg(windows)]
    fn join_paths_windows() {
        fn test_eq(input: &[&str], output: &str) -> bool {
            &*join_paths(input.iter().cloned()).unwrap() ==
                OsStr::from_str(output)
        }

        assert!(test_eq(&[], ""));
        assert!(test_eq(&[r"c:\windows", r"c:\"],
                        r"c:\windows;c:\"));
        assert!(test_eq(&["", r"c:\windows", "", "", r"c:\", ""],
                        r";c:\windows;;;c:\;"));
        assert!(test_eq(&[r"c:\te;st", r"c:\"],
                        r#""c:\te;st";c:\"#));
        assert!(join_paths([r#"c:\te"st"#].iter().cloned()).is_err());
    }
    }
