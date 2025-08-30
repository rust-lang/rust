//! Inspection and manipulation of the process's environment.
//!
//! This module contains functions to inspect various aspects such as
//! environment variables, process arguments, the current directory, and various
//! other important directories.
//!
//! There are several functions and structs in this module that have a
//! counterpart ending in `os`. Those ending in `os` will return an [`OsString`]
//! and those without will return a [`String`].

#![stable(feature = "env", since = "1.0.0")]

use crate::error::Error;
use crate::ffi::{OsStr, OsString};
use crate::num::NonZero;
use crate::ops::Try;
use crate::path::{Path, PathBuf};
use crate::sys::{env as env_imp, os as os_imp};
use crate::{array, fmt, io, sys};

/// Returns the current working directory as a [`PathBuf`].
///
/// # Platform-specific behavior
///
/// This function [currently] corresponds to the `getcwd` function on Unix
/// and the `GetCurrentDirectoryW` function on Windows.
///
/// [currently]: crate::io#platform-specific-behavior
///
/// # Errors
///
/// Returns an [`Err`] if the current working directory value is invalid.
/// Possible cases:
///
/// * Current directory does not exist.
/// * There are insufficient permissions to access the current directory.
///
/// # Examples
///
/// ```
/// use std::env;
///
/// fn main() -> std::io::Result<()> {
///     let path = env::current_dir()?;
///     println!("The current directory is {}", path.display());
///     Ok(())
/// }
/// ```
#[doc(alias = "pwd")]
#[doc(alias = "getcwd")]
#[doc(alias = "GetCurrentDirectory")]
#[stable(feature = "env", since = "1.0.0")]
pub fn current_dir() -> io::Result<PathBuf> {
    os_imp::getcwd()
}

/// Changes the current working directory to the specified path.
///
/// # Platform-specific behavior
///
/// This function [currently] corresponds to the `chdir` function on Unix
/// and the `SetCurrentDirectoryW` function on Windows.
///
/// Returns an [`Err`] if the operation fails.
///
/// [currently]: crate::io#platform-specific-behavior
///
/// # Examples
///
/// ```
/// use std::env;
/// use std::path::Path;
///
/// let root = Path::new("/");
/// assert!(env::set_current_dir(&root).is_ok());
/// println!("Successfully changed working directory to {}!", root.display());
/// ```
#[doc(alias = "chdir", alias = "SetCurrentDirectory", alias = "SetCurrentDirectoryW")]
#[stable(feature = "env", since = "1.0.0")]
pub fn set_current_dir<P: AsRef<Path>>(path: P) -> io::Result<()> {
    os_imp::chdir(path.as_ref())
}

/// An iterator over a snapshot of the environment variables of this process.
///
/// This structure is created by [`env::vars()`]. See its documentation for more.
///
/// [`env::vars()`]: vars
#[stable(feature = "env", since = "1.0.0")]
pub struct Vars {
    inner: VarsOs,
}

/// An iterator over a snapshot of the environment variables of this process.
///
/// This structure is created by [`env::vars_os()`]. See its documentation for more.
///
/// [`env::vars_os()`]: vars_os
#[stable(feature = "env", since = "1.0.0")]
pub struct VarsOs {
    inner: env_imp::Env,
}

/// Returns an iterator of (variable, value) pairs of strings, for all the
/// environment variables of the current process.
///
/// The returned iterator contains a snapshot of the process's environment
/// variables at the time of this invocation. Modifications to environment
/// variables afterwards will not be reflected in the returned iterator.
///
/// # Panics
///
/// While iterating, the returned iterator will panic if any key or value in the
/// environment is not valid unicode. If this is not desired, consider using
/// [`env::vars_os()`].
///
/// # Examples
///
/// ```
/// // Print all environment variables.
/// for (key, value) in std::env::vars() {
///     println!("{key}: {value}");
/// }
/// ```
///
/// [`env::vars_os()`]: vars_os
#[must_use]
#[stable(feature = "env", since = "1.0.0")]
pub fn vars() -> Vars {
    Vars { inner: vars_os() }
}

/// Returns an iterator of (variable, value) pairs of OS strings, for all the
/// environment variables of the current process.
///
/// The returned iterator contains a snapshot of the process's environment
/// variables at the time of this invocation. Modifications to environment
/// variables afterwards will not be reflected in the returned iterator.
///
/// Note that the returned iterator will not check if the environment variables
/// are valid Unicode. If you want to panic on invalid UTF-8,
/// use the [`vars`] function instead.
///
/// # Examples
///
/// ```
/// // Print all environment variables.
/// for (key, value) in std::env::vars_os() {
///     println!("{key:?}: {value:?}");
/// }
/// ```
#[must_use]
#[stable(feature = "env", since = "1.0.0")]
pub fn vars_os() -> VarsOs {
    VarsOs { inner: env_imp::env() }
}

#[stable(feature = "env", since = "1.0.0")]
impl Iterator for Vars {
    type Item = (String, String);
    fn next(&mut self) -> Option<(String, String)> {
        self.inner.next().map(|(a, b)| (a.into_string().unwrap(), b.into_string().unwrap()))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Vars {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { inner: VarsOs { inner } } = self;
        f.debug_struct("Vars").field("inner", &inner.str_debug()).finish()
    }
}

#[stable(feature = "env", since = "1.0.0")]
impl Iterator for VarsOs {
    type Item = (OsString, OsString);
    fn next(&mut self) -> Option<(OsString, OsString)> {
        self.inner.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for VarsOs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { inner } = self;
        f.debug_struct("VarsOs").field("inner", inner).finish()
    }
}

/// Fetches the environment variable `key` from the current process.
///
/// # Errors
///
/// Returns [`VarError::NotPresent`] if:
/// - The variable is not set.
/// - The variable's name contains an equal sign or NUL (`'='` or `'\0'`).
///
/// Returns [`VarError::NotUnicode`] if the variable's value is not valid
/// Unicode. If this is not desired, consider using [`var_os`].
///
/// Use [`env!`] or [`option_env!`] instead if you want to check environment
/// variables at compile time.
///
/// # Examples
///
/// ```
/// use std::env;
///
/// let key = "HOME";
/// match env::var(key) {
///     Ok(val) => println!("{key}: {val:?}"),
///     Err(e) => println!("couldn't interpret {key}: {e}"),
/// }
/// ```
#[stable(feature = "env", since = "1.0.0")]
pub fn var<K: AsRef<OsStr>>(key: K) -> Result<String, VarError> {
    _var(key.as_ref())
}

fn _var(key: &OsStr) -> Result<String, VarError> {
    match var_os(key) {
        Some(s) => s.into_string().map_err(VarError::NotUnicode),
        None => Err(VarError::NotPresent),
    }
}

/// Fetches the environment variable `key` from the current process, returning
/// [`None`] if the variable isn't set or if there is another error.
///
/// It may return `None` if the environment variable's name contains
/// the equal sign character (`=`) or the NUL character.
///
/// Note that this function will not check if the environment variable
/// is valid Unicode. If you want to have an error on invalid UTF-8,
/// use the [`var`] function instead.
///
/// # Examples
///
/// ```
/// use std::env;
///
/// let key = "HOME";
/// match env::var_os(key) {
///     Some(val) => println!("{key}: {val:?}"),
///     None => println!("{key} is not defined in the environment.")
/// }
/// ```
///
/// If expecting a delimited variable (such as `PATH`), [`split_paths`]
/// can be used to separate items.
#[must_use]
#[stable(feature = "env", since = "1.0.0")]
pub fn var_os<K: AsRef<OsStr>>(key: K) -> Option<OsString> {
    _var_os(key.as_ref())
}

fn _var_os(key: &OsStr) -> Option<OsString> {
    env_imp::getenv(key)
}

/// The error type for operations interacting with environment variables.
/// Possibly returned from [`env::var()`].
///
/// [`env::var()`]: var
#[derive(Debug, PartialEq, Eq, Clone)]
#[stable(feature = "env", since = "1.0.0")]
pub enum VarError {
    /// The specified environment variable was not present in the current
    /// process's environment.
    #[stable(feature = "env", since = "1.0.0")]
    NotPresent,

    /// The specified environment variable was found, but it did not contain
    /// valid unicode data. The found data is returned as a payload of this
    /// variant.
    #[stable(feature = "env", since = "1.0.0")]
    NotUnicode(#[stable(feature = "env", since = "1.0.0")] OsString),
}

#[stable(feature = "env", since = "1.0.0")]
impl fmt::Display for VarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            VarError::NotPresent => write!(f, "environment variable not found"),
            VarError::NotUnicode(ref s) => {
                write!(f, "environment variable was not valid unicode: {:?}", s)
            }
        }
    }
}

#[stable(feature = "env", since = "1.0.0")]
impl Error for VarError {}

/// Sets the environment variable `key` to the value `value` for the currently running
/// process.
///
/// # Safety
///
/// This function is safe to call in a single-threaded program.
///
/// This function is also always safe to call on Windows, in single-threaded
/// and multi-threaded programs.
///
/// In multi-threaded programs on other operating systems, the only safe option is
/// to not use `set_var` or `remove_var` at all.
///
/// The exact requirement is: you
/// must ensure that there are no other threads concurrently writing or
/// *reading*(!) the environment through functions or global variables other
/// than the ones in this module. The problem is that these operating systems
/// do not provide a thread-safe way to read the environment, and most C
/// libraries, including libc itself, do not advertise which functions read
/// from the environment. Even functions from the Rust standard library may
/// read the environment without going through this module, e.g. for DNS
/// lookups from [`std::net::ToSocketAddrs`]. No stable guarantee is made about
/// which functions may read from the environment in future versions of a
/// library. All this makes it not practically possible for you to guarantee
/// that no other thread will read the environment, so the only safe option is
/// to not use `set_var` or `remove_var` in multi-threaded programs at all.
///
/// Discussion of this unsafety on Unix may be found in:
///
///  - [Austin Group Bugzilla (for POSIX)](https://austingroupbugs.net/view.php?id=188)
///  - [GNU C library Bugzilla](https://sourceware.org/bugzilla/show_bug.cgi?id=15607#c2)
///
/// To pass an environment variable to a child process, you can instead use [`Command::env`].
///
/// [`std::net::ToSocketAddrs`]: crate::net::ToSocketAddrs
/// [`Command::env`]: crate::process::Command::env
///
/// # Panics
///
/// This function may panic if `key` is empty, contains an ASCII equals sign `'='`
/// or the NUL character `'\0'`, or when `value` contains the NUL character.
///
/// # Examples
///
/// ```
/// use std::env;
///
/// let key = "KEY";
/// unsafe {
///     env::set_var(key, "VALUE");
/// }
/// assert_eq!(env::var(key), Ok("VALUE".to_string()));
/// ```
#[rustc_deprecated_safe_2024(
    audit_that = "the environment access only happens in single-threaded code"
)]
#[stable(feature = "env", since = "1.0.0")]
pub unsafe fn set_var<K: AsRef<OsStr>, V: AsRef<OsStr>>(key: K, value: V) {
    let (key, value) = (key.as_ref(), value.as_ref());
    unsafe { env_imp::setenv(key, value) }.unwrap_or_else(|e| {
        panic!("failed to set environment variable `{key:?}` to `{value:?}`: {e}")
    })
}

/// Removes an environment variable from the environment of the currently running process.
///
/// # Safety
///
/// This function is safe to call in a single-threaded program.
///
/// This function is also always safe to call on Windows, in single-threaded
/// and multi-threaded programs.
///
/// In multi-threaded programs on other operating systems, the only safe option is
/// to not use `set_var` or `remove_var` at all.
///
/// The exact requirement is: you
/// must ensure that there are no other threads concurrently writing or
/// *reading*(!) the environment through functions or global variables other
/// than the ones in this module. The problem is that these operating systems
/// do not provide a thread-safe way to read the environment, and most C
/// libraries, including libc itself, do not advertise which functions read
/// from the environment. Even functions from the Rust standard library may
/// read the environment without going through this module, e.g. for DNS
/// lookups from [`std::net::ToSocketAddrs`]. No stable guarantee is made about
/// which functions may read from the environment in future versions of a
/// library. All this makes it not practically possible for you to guarantee
/// that no other thread will read the environment, so the only safe option is
/// to not use `set_var` or `remove_var` in multi-threaded programs at all.
///
/// Discussion of this unsafety on Unix may be found in:
///
///  - [Austin Group Bugzilla](https://austingroupbugs.net/view.php?id=188)
///  - [GNU C library Bugzilla](https://sourceware.org/bugzilla/show_bug.cgi?id=15607#c2)
///
/// To prevent a child process from inheriting an environment variable, you can
/// instead use [`Command::env_remove`] or [`Command::env_clear`].
///
/// [`std::net::ToSocketAddrs`]: crate::net::ToSocketAddrs
/// [`Command::env_remove`]: crate::process::Command::env_remove
/// [`Command::env_clear`]: crate::process::Command::env_clear
///
/// # Panics
///
/// This function may panic if `key` is empty, contains an ASCII equals sign
/// `'='` or the NUL character `'\0'`, or when the value contains the NUL
/// character.
///
/// # Examples
///
/// ```no_run
/// use std::env;
///
/// let key = "KEY";
/// unsafe {
///     env::set_var(key, "VALUE");
/// }
/// assert_eq!(env::var(key), Ok("VALUE".to_string()));
///
/// unsafe {
///     env::remove_var(key);
/// }
/// assert!(env::var(key).is_err());
/// ```
#[rustc_deprecated_safe_2024(
    audit_that = "the environment access only happens in single-threaded code"
)]
#[stable(feature = "env", since = "1.0.0")]
pub unsafe fn remove_var<K: AsRef<OsStr>>(key: K) {
    let key = key.as_ref();
    unsafe { env_imp::unsetenv(key) }
        .unwrap_or_else(|e| panic!("failed to remove environment variable `{key:?}`: {e}"))
}

/// An iterator that splits an environment variable into paths according to
/// platform-specific conventions.
///
/// The iterator element type is [`PathBuf`].
///
/// This structure is created by [`env::split_paths()`]. See its
/// documentation for more.
///
/// [`env::split_paths()`]: split_paths
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "env", since = "1.0.0")]
pub struct SplitPaths<'a> {
    inner: os_imp::SplitPaths<'a>,
}

/// Parses input according to platform conventions for the `PATH`
/// environment variable.
///
/// Returns an iterator over the paths contained in `unparsed`. The iterator
/// element type is [`PathBuf`].
///
/// On most Unix platforms, the separator is `:` and on Windows it is `;`. This
/// also performs unquoting on Windows.
///
/// [`join_paths`] can be used to recombine elements.
///
/// # Panics
///
/// This will panic on systems where there is no delimited `PATH` variable,
/// such as UEFI.
///
/// # Examples
///
/// ```
/// use std::env;
///
/// let key = "PATH";
/// match env::var_os(key) {
///     Some(paths) => {
///         for path in env::split_paths(&paths) {
///             println!("'{}'", path.display());
///         }
///     }
///     None => println!("{key} is not defined in the environment.")
/// }
/// ```
#[stable(feature = "env", since = "1.0.0")]
pub fn split_paths<T: AsRef<OsStr> + ?Sized>(unparsed: &T) -> SplitPaths<'_> {
    SplitPaths { inner: os_imp::split_paths(unparsed.as_ref()) }
}

#[stable(feature = "env", since = "1.0.0")]
impl<'a> Iterator for SplitPaths<'a> {
    type Item = PathBuf;
    fn next(&mut self) -> Option<PathBuf> {
        self.inner.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for SplitPaths<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplitPaths").finish_non_exhaustive()
    }
}

/// The error type for operations on the `PATH` variable. Possibly returned from
/// [`env::join_paths()`].
///
/// [`env::join_paths()`]: join_paths
#[derive(Debug)]
#[stable(feature = "env", since = "1.0.0")]
pub struct JoinPathsError {
    inner: os_imp::JoinPathsError,
}

/// Joins a collection of [`Path`]s appropriately for the `PATH`
/// environment variable.
///
/// # Errors
///
/// Returns an [`Err`] (containing an error message) if one of the input
/// [`Path`]s contains an invalid character for constructing the `PATH`
/// variable (a double quote on Windows or a colon on Unix), or if the system
/// does not have a `PATH`-like variable (e.g. UEFI or WASI).
///
/// # Examples
///
/// Joining paths on a Unix-like platform:
///
/// ```
/// use std::env;
/// use std::ffi::OsString;
/// use std::path::Path;
///
/// fn main() -> Result<(), env::JoinPathsError> {
/// # if cfg!(unix) {
///     let paths = [Path::new("/bin"), Path::new("/usr/bin")];
///     let path_os_string = env::join_paths(paths.iter())?;
///     assert_eq!(path_os_string, OsString::from("/bin:/usr/bin"));
/// # }
///     Ok(())
/// }
/// ```
///
/// Joining a path containing a colon on a Unix-like platform results in an
/// error:
///
/// ```
/// # if cfg!(unix) {
/// use std::env;
/// use std::path::Path;
///
/// let paths = [Path::new("/bin"), Path::new("/usr/bi:n")];
/// assert!(env::join_paths(paths.iter()).is_err());
/// # }
/// ```
///
/// Using `env::join_paths()` with [`env::split_paths()`] to append an item to
/// the `PATH` environment variable:
///
/// ```
/// use std::env;
/// use std::path::PathBuf;
///
/// fn main() -> Result<(), env::JoinPathsError> {
///     if let Some(path) = env::var_os("PATH") {
///         let mut paths = env::split_paths(&path).collect::<Vec<_>>();
///         paths.push(PathBuf::from("/home/xyz/bin"));
///         let new_path = env::join_paths(paths)?;
///         unsafe { env::set_var("PATH", &new_path); }
///     }
///
///     Ok(())
/// }
/// ```
///
/// [`env::split_paths()`]: split_paths
#[stable(feature = "env", since = "1.0.0")]
pub fn join_paths<I, T>(paths: I) -> Result<OsString, JoinPathsError>
where
    I: IntoIterator<Item = T>,
    T: AsRef<OsStr>,
{
    os_imp::join_paths(paths.into_iter()).map_err(|e| JoinPathsError { inner: e })
}

#[stable(feature = "env", since = "1.0.0")]
impl fmt::Display for JoinPathsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

#[stable(feature = "env", since = "1.0.0")]
impl Error for JoinPathsError {
    #[allow(deprecated, deprecated_in_future)]
    fn description(&self) -> &str {
        self.inner.description()
    }
}

/// Returns the path of the current user's home directory if known.
///
/// This may return `None` if getting the directory fails or if the platform does not have user home directories.
///
/// For storing user data and configuration it is often preferable to use more specific directories.
/// For example, [XDG Base Directories] on Unix or the `LOCALAPPDATA` and `APPDATA` environment variables on Windows.
///
/// [XDG Base Directories]: https://specifications.freedesktop.org/basedir-spec/latest/
///
/// # Unix
///
/// - Returns the value of the 'HOME' environment variable if it is set
///   (and not an empty string).
/// - Otherwise, it tries to determine the home directory by invoking the `getpwuid_r` function
///   using the UID of the current user. An empty home directory field returned from the
///   `getpwuid_r` function is considered to be a valid value.
/// - Returns `None` if the current user has no entry in the /etc/passwd file.
///
/// # Windows
///
/// - Returns the value of the 'USERPROFILE' environment variable if it is set, and is not an empty string.
/// - Otherwise, [`GetUserProfileDirectory`][msdn] is used to return the path. This may change in the future.
///
/// [msdn]: https://docs.microsoft.com/en-us/windows/win32/api/userenv/nf-userenv-getuserprofiledirectorya
///
/// In UWP (Universal Windows Platform) targets this function is unimplemented and always returns `None`.
///
/// Before Rust 1.85.0, this function used to return the value of the 'HOME' environment variable
/// on Windows, which in Cygwin or Mingw environments could return non-standard paths like `/home/you`
/// instead of `C:\Users\you`.
///
/// # Examples
///
/// ```
/// use std::env;
///
/// match env::home_dir() {
///     Some(path) => println!("Your home directory, probably: {}", path.display()),
///     None => println!("Impossible to get your home dir!"),
/// }
/// ```
#[must_use]
#[stable(feature = "env", since = "1.0.0")]
pub fn home_dir() -> Option<PathBuf> {
    os_imp::home_dir()
}

/// Returns the path of a temporary directory.
///
/// The temporary directory may be shared among users, or between processes
/// with different privileges; thus, the creation of any files or directories
/// in the temporary directory must use a secure method to create a uniquely
/// named file. Creating a file or directory with a fixed or predictable name
/// may result in "insecure temporary file" security vulnerabilities. Consider
/// using a crate that securely creates temporary files or directories.
///
/// Note that the returned value may be a symbolic link, not a directory.
///
/// # Platform-specific behavior
///
/// On Unix, returns the value of the `TMPDIR` environment variable if it is
/// set, otherwise the value is OS-specific:
/// - On Android, there is no global temporary folder (it is usually allocated
///   per-app), it will return the application's cache dir if the program runs
///   in application's namespace and system version is Android 13 (or above), or
///   `/data/local/tmp` otherwise.
/// - On Darwin-based OSes (macOS, iOS, etc) it returns the directory provided
///   by `confstr(_CS_DARWIN_USER_TEMP_DIR, ...)`, as recommended by [Apple's
///   security guidelines][appledoc].
/// - On all other unix-based OSes, it returns `/tmp`.
///
/// On Windows, the behavior is equivalent to that of [`GetTempPath2`][GetTempPath2] /
/// [`GetTempPath`][GetTempPath], which this function uses internally.
///
/// Note that, this [may change in the future][changes].
///
/// [changes]: io#platform-specific-behavior
/// [GetTempPath2]: https://docs.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-gettemppath2a
/// [GetTempPath]: https://docs.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-gettemppatha
/// [appledoc]: https://developer.apple.com/library/archive/documentation/Security/Conceptual/SecureCodingGuide/Articles/RaceConditions.html#//apple_ref/doc/uid/TP40002585-SW10
///
/// ```no_run
/// use std::env;
///
/// fn main() {
///     let dir = env::temp_dir();
///     println!("Temporary directory: {}", dir.display());
/// }
/// ```
#[must_use]
#[doc(alias = "GetTempPath", alias = "GetTempPath2")]
#[stable(feature = "env", since = "1.0.0")]
pub fn temp_dir() -> PathBuf {
    os_imp::temp_dir()
}

/// Returns the full filesystem path of the current running executable.
///
/// # Platform-specific behavior
///
/// If the executable was invoked through a symbolic link, some platforms will
/// return the path of the symbolic link and other platforms will return the
/// path of the symbolic link’s target.
///
/// If the executable is renamed while it is running, platforms may return the
/// path at the time it was loaded instead of the new path.
///
/// # Errors
///
/// Acquiring the path of the current executable is a platform-specific operation
/// that can fail for a good number of reasons. Some errors can include, but not
/// be limited to, filesystem operations failing or general syscall failures.
///
/// # Security
///
/// The output of this function should not be trusted for anything
/// that might have security implications. Basically, if users can run
/// the executable, they can change the output arbitrarily.
///
/// As an example, you can easily introduce a race condition. It goes
/// like this:
///
/// 1. You get the path to the current executable using `current_exe()`, and
///    store it in a variable.
/// 2. Time passes. A malicious actor removes the current executable, and
///    replaces it with a malicious one.
/// 3. You then use the stored path to re-execute the current
///    executable.
///
/// You expected to safely execute the current executable, but you're
/// instead executing something completely different. The code you
/// just executed run with your privileges.
///
/// This sort of behavior has been known to [lead to privilege escalation] when
/// used incorrectly.
///
/// [lead to privilege escalation]: https://securityvulns.com/Wdocument183.html
///
/// # Examples
///
/// ```
/// use std::env;
///
/// match env::current_exe() {
///     Ok(exe_path) => println!("Path of this executable is: {}",
///                              exe_path.display()),
///     Err(e) => println!("failed to get current exe path: {e}"),
/// };
/// ```
#[stable(feature = "env", since = "1.0.0")]
pub fn current_exe() -> io::Result<PathBuf> {
    os_imp::current_exe()
}

/// An iterator over the arguments of a process, yielding a [`String`] value for
/// each argument.
///
/// This struct is created by [`env::args()`]. See its documentation
/// for more.
///
/// The first element is traditionally the path of the executable, but it can be
/// set to arbitrary text, and might not even exist. This means this property
/// should not be relied upon for security purposes.
///
/// [`env::args()`]: args
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "env", since = "1.0.0")]
pub struct Args {
    inner: ArgsOs,
}

/// An iterator over the arguments of a process, yielding an [`OsString`] value
/// for each argument.
///
/// This struct is created by [`env::args_os()`]. See its documentation
/// for more.
///
/// The first element is traditionally the path of the executable, but it can be
/// set to arbitrary text, and might not even exist. This means this property
/// should not be relied upon for security purposes.
///
/// [`env::args_os()`]: args_os
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "env", since = "1.0.0")]
pub struct ArgsOs {
    inner: sys::args::Args,
}

/// Returns the arguments that this program was started with (normally passed
/// via the command line).
///
/// The first element is traditionally the path of the executable, but it can be
/// set to arbitrary text, and might not even exist. This means this property should
/// not be relied upon for security purposes.
///
/// On Unix systems the shell usually expands unquoted arguments with glob patterns
/// (such as `*` and `?`). On Windows this is not done, and such arguments are
/// passed as-is.
///
/// On glibc Linux systems, arguments are retrieved by placing a function in `.init_array`.
/// glibc passes `argc`, `argv`, and `envp` to functions in `.init_array`, as a non-standard
/// extension. This allows `std::env::args` to work even in a `cdylib` or `staticlib`, as it
/// does on macOS and Windows.
///
/// # Panics
///
/// The returned iterator will panic during iteration if any argument to the
/// process is not valid Unicode. If this is not desired,
/// use the [`args_os`] function instead.
///
/// # Examples
///
/// ```
/// use std::env;
///
/// // Prints each argument on a separate line
/// for argument in env::args() {
///     println!("{argument}");
/// }
/// ```
#[stable(feature = "env", since = "1.0.0")]
pub fn args() -> Args {
    Args { inner: args_os() }
}

/// Returns the arguments that this program was started with (normally passed
/// via the command line).
///
/// The first element is traditionally the path of the executable, but it can be
/// set to arbitrary text, and might not even exist. This means this property should
/// not be relied upon for security purposes.
///
/// On Unix systems the shell usually expands unquoted arguments with glob patterns
/// (such as `*` and `?`). On Windows this is not done, and such arguments are
/// passed as-is.
///
/// On glibc Linux systems, arguments are retrieved by placing a function in `.init_array`.
/// glibc passes `argc`, `argv`, and `envp` to functions in `.init_array`, as a non-standard
/// extension. This allows `std::env::args_os` to work even in a `cdylib` or `staticlib`, as it
/// does on macOS and Windows.
///
/// Note that the returned iterator will not check if the arguments to the
/// process are valid Unicode. If you want to panic on invalid UTF-8,
/// use the [`args`] function instead.
///
/// # Examples
///
/// ```
/// use std::env;
///
/// // Prints each argument on a separate line
/// for argument in env::args_os() {
///     println!("{argument:?}");
/// }
/// ```
#[stable(feature = "env", since = "1.0.0")]
pub fn args_os() -> ArgsOs {
    ArgsOs { inner: sys::args::args() }
}

#[stable(feature = "env_unimpl_send_sync", since = "1.26.0")]
impl !Send for Args {}

#[stable(feature = "env_unimpl_send_sync", since = "1.26.0")]
impl !Sync for Args {}

#[stable(feature = "env", since = "1.0.0")]
impl Iterator for Args {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        self.inner.next().map(|s| s.into_string().unwrap())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    // Methods which skip args cannot simply delegate to the inner iterator,
    // because `env::args` states that we will "panic during iteration if any
    // argument to the process is not valid Unicode".
    //
    // This offers two possible interpretations:
    // - a skipped argument is never encountered "during iteration"
    // - even a skipped argument is encountered "during iteration"
    //
    // As a panic can be observed, we err towards validating even skipped
    // arguments for now, though this is not explicitly promised by the API.
}

#[stable(feature = "env", since = "1.0.0")]
impl ExactSizeIterator for Args {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[stable(feature = "env_iterators", since = "1.12.0")]
impl DoubleEndedIterator for Args {
    fn next_back(&mut self) -> Option<String> {
        self.inner.next_back().map(|s| s.into_string().unwrap())
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { inner: ArgsOs { inner } } = self;
        f.debug_struct("Args").field("inner", inner).finish()
    }
}

#[stable(feature = "env_unimpl_send_sync", since = "1.26.0")]
impl !Send for ArgsOs {}

#[stable(feature = "env_unimpl_send_sync", since = "1.26.0")]
impl !Sync for ArgsOs {}

#[stable(feature = "env", since = "1.0.0")]
impl Iterator for ArgsOs {
    type Item = OsString;

    #[inline]
    fn next(&mut self) -> Option<OsString> {
        self.inner.next()
    }

    #[inline]
    fn next_chunk<const N: usize>(
        &mut self,
    ) -> Result<[OsString; N], array::IntoIter<OsString, N>> {
        self.inner.next_chunk()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.inner.len()
    }

    #[inline]
    fn last(self) -> Option<OsString> {
        self.inner.last()
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.inner.advance_by(n)
    }

    #[inline]
    fn try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        self.inner.try_fold(init, f)
    }

    #[inline]
    fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        self.inner.fold(init, f)
    }
}

#[stable(feature = "env", since = "1.0.0")]
impl ExactSizeIterator for ArgsOs {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[stable(feature = "env_iterators", since = "1.12.0")]
impl DoubleEndedIterator for ArgsOs {
    #[inline]
    fn next_back(&mut self) -> Option<OsString> {
        self.inner.next_back()
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.inner.advance_back_by(n)
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for ArgsOs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { inner } = self;
        f.debug_struct("ArgsOs").field("inner", inner).finish()
    }
}

/// Constants associated with the current target
#[stable(feature = "env", since = "1.0.0")]
pub mod consts {
    use crate::sys::env_consts::os;

    /// A string describing the architecture of the CPU that is currently in use.
    /// An example value may be: `"x86"`, `"arm"` or `"riscv64"`.
    ///
    /// <details><summary>Full list of possible values</summary>
    ///
    /// * `"x86"`
    /// * `"x86_64"`
    /// * `"arm"`
    /// * `"aarch64"`
    /// * `"m68k"`
    /// * `"mips"`
    /// * `"mips32r6"`
    /// * `"mips64"`
    /// * `"mips64r6"`
    /// * `"csky"`
    /// * `"powerpc"`
    /// * `"powerpc64"`
    /// * `"riscv32"`
    /// * `"riscv64"`
    /// * `"s390x"`
    /// * `"sparc"`
    /// * `"sparc64"`
    /// * `"hexagon"`
    /// * `"loongarch32"`
    /// * `"loongarch64"`
    ///
    /// </details>
    #[stable(feature = "env", since = "1.0.0")]
    pub const ARCH: &str = env!("STD_ENV_ARCH");

    /// A string describing the family of the operating system.
    /// An example value may be: `"unix"`, or `"windows"`.
    ///
    /// This value may be an empty string if the family is unknown.
    ///
    /// <details><summary>Full list of possible values</summary>
    ///
    /// * `"unix"`
    /// * `"windows"`
    /// * `"itron"`
    /// * `"wasm"`
    /// * `""`
    ///
    /// </details>
    #[stable(feature = "env", since = "1.0.0")]
    pub const FAMILY: &str = os::FAMILY;

    /// A string describing the specific operating system in use.
    /// An example value may be: `"linux"`, or `"freebsd"`.
    ///
    /// <details><summary>Full list of possible values</summary>
    ///
    /// * `"linux"`
    /// * `"windows"`
    /// * `"macos"`
    /// * `"android"`
    /// * `"ios"`
    /// * `"openbsd"`
    /// * `"freebsd"`
    /// * `"netbsd"`
    /// * `"wasi"`
    /// * `"hermit"`
    /// * `"aix"`
    /// * `"apple"`
    /// * `"dragonfly"`
    /// * `"emscripten"`
    /// * `"espidf"`
    /// * `"fortanix"`
    /// * `"uefi"`
    /// * `"fuchsia"`
    /// * `"haiku"`
    /// * `"hermit"`
    /// * `"watchos"`
    /// * `"visionos"`
    /// * `"tvos"`
    /// * `"horizon"`
    /// * `"hurd"`
    /// * `"illumos"`
    /// * `"l4re"`
    /// * `"nto"`
    /// * `"redox"`
    /// * `"solaris"`
    /// * `"solid_asp3`
    /// * `"vexos"`
    /// * `"vita"`
    /// * `"vxworks"`
    /// * `"xous"`
    ///
    /// </details>
    #[stable(feature = "env", since = "1.0.0")]
    pub const OS: &str = os::OS;

    /// Specifies the filename prefix, if any, used for shared libraries on this platform.
    /// This is either `"lib"` or an empty string. (`""`).
    #[stable(feature = "env", since = "1.0.0")]
    pub const DLL_PREFIX: &str = os::DLL_PREFIX;

    /// Specifies the filename suffix, if any, used for shared libraries on this platform.
    /// An example value may be: `".so"`, `".elf"`, or `".dll"`.
    ///
    /// The possible values are identical to those of [`DLL_EXTENSION`], but with the leading period included.
    #[stable(feature = "env", since = "1.0.0")]
    pub const DLL_SUFFIX: &str = os::DLL_SUFFIX;

    /// Specifies the file extension, if any, used for shared libraries on this platform that goes after the dot.
    /// An example value may be: `"so"`, `"elf"`, or `"dll"`.
    ///
    /// <details><summary>Full list of possible values</summary>
    ///
    /// * `"so"`
    /// * `"dylib"`
    /// * `"dll"`
    /// * `"sgxs"`
    /// * `"a"`
    /// * `"elf"`
    /// * `"wasm"`
    /// * `""` (an empty string)
    ///
    /// </details>
    #[stable(feature = "env", since = "1.0.0")]
    pub const DLL_EXTENSION: &str = os::DLL_EXTENSION;

    /// Specifies the filename suffix, if any, used for executable binaries on this platform.
    /// An example value may be: `".exe"`, or `".efi"`.
    ///
    /// The possible values are identical to those of [`EXE_EXTENSION`], but with the leading period included.
    #[stable(feature = "env", since = "1.0.0")]
    pub const EXE_SUFFIX: &str = os::EXE_SUFFIX;

    /// Specifies the file extension, if any, used for executable binaries on this platform.
    /// An example value may be: `"exe"`, or an empty string (`""`).
    ///
    /// <details><summary>Full list of possible values</summary>
    ///
    /// * `"bin"`
    /// * `"exe"`
    /// * `"efi"`
    /// * `"js"`
    /// * `"sgxs"`
    /// * `"elf"`
    /// * `"wasm"`
    /// * `""` (an empty string)
    ///
    /// </details>
    #[stable(feature = "env", since = "1.0.0")]
    pub const EXE_EXTENSION: &str = os::EXE_EXTENSION;
}
