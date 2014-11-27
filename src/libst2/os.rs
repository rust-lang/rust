// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Higher-level interfaces to libc::* functions and operating system services.
 *
 * In general these take and return rust types, use rust idioms (enums,
 * closures, vectors) rather than C idioms, and do more extensive safety
 * checks.
 *
 * This module is not meant to only contain 1:1 mappings to libc entries; any
 * os-interface code that is reasonably useful and broadly applicable can go
 * here. Including utility routines that merely build on other os code.
 *
 * We assume the general case is that users do not care, and do not want to
 * be made to care, which operating system they are on. While they may want
 * to special case various special cases -- and so we will not _hide_ the
 * facts of which OS the user is on -- they should be given the opportunity
 * to write OS-ignorant code by default.
 */

#![experimental]

#![allow(missing_docs)]
#![allow(non_snake_case)]

pub use self::MemoryMapKind::*;
pub use self::MapOption::*;
pub use self::MapError::*;

use clone::Clone;
use error::{FromError, Error};
use fmt;
use io::{IoResult, IoError};
use iter::Iterator;
use libc::{c_void, c_int};
use libc;
use boxed::Box;
use ops::Drop;
use option::{Some, None, Option};
use os;
use path::{Path, GenericPath, BytesContainer};
use sys;
use sys::os as os_imp;
use ptr::RawPtr;
use ptr;
use result::{Err, Ok, Result};
use slice::{AsSlice, SlicePrelude, PartialEqSlicePrelude};
use slice::CloneSliceAllocPrelude;
use str::{Str, StrPrelude, StrAllocating};
use string::{String, ToString};
use sync::atomic::{AtomicInt, INIT_ATOMIC_INT, SeqCst};
use vec::Vec;

#[cfg(unix)] use c_str::ToCStr;
#[cfg(unix)] use libc::c_char;

/// Get the number of cores available
pub fn num_cpus() -> uint { unimplemented!() }

pub const TMPBUF_SZ : uint = 1000u;
const BUF_BYTES : uint = 2048u;

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
/// use std::os;
///
/// // We assume that we are in a valid directory like "/home".
/// let current_working_directory = os::getcwd().unwrap();
/// println!("The current directory is {}", current_working_directory.display());
/// // /home
/// ```
#[cfg(unix)]
pub fn getcwd() -> IoResult<Path> { unimplemented!() }

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
/// use std::os;
///
/// // We assume that we are in a valid directory like "C:\\Windows".
/// let current_working_directory = os::getcwd().unwrap();
/// println!("The current directory is {}", current_working_directory.display());
/// // C:\\Windows
/// ```
#[cfg(windows)]
pub fn getcwd() -> IoResult<Path> { unimplemented!() }

#[cfg(windows)]
pub mod windows {
    use libc::types::os::arch::extra::DWORD;
    use libc;
    use option::{None, Option};
    use option;
    use os::TMPBUF_SZ;
    use slice::{SlicePrelude};
    use string::String;
    use str::StrPrelude;
    use vec::Vec;

    pub fn fill_utf16_buf_and_decode(f: |*mut u16, DWORD| -> DWORD)
        -> Option<String> { unimplemented!() }
}

/*
Accessing environment variables is not generally threadsafe.
Serialize access through a global lock.
*/
fn with_env_lock<T>(f: || -> T) -> T { unimplemented!() }

/// Returns a vector of (variable, value) pairs, for all the environment
/// variables of the current process.
///
/// Invalid UTF-8 bytes are replaced with \uFFFD. See `String::from_utf8_lossy()`
/// for details.
///
/// # Example
///
/// ```rust
/// use std::os;
///
/// // We will iterate through the references to the element returned by os::env();
/// for &(ref key, ref value) in os::env().iter() {
///     println!("'{}': '{}'", key, value );
/// }
/// ```
pub fn env() -> Vec<(String,String)> { unimplemented!() }

/// Returns a vector of (variable, value) byte-vector pairs for all the
/// environment variables of the current process.
pub fn env_as_bytes() -> Vec<(Vec<u8>,Vec<u8>)> { unimplemented!() }

#[cfg(unix)]
/// Fetches the environment variable `n` from the current process, returning
/// None if the variable isn't set.
///
/// Any invalid UTF-8 bytes in the value are replaced by \uFFFD. See
/// `String::from_utf8_lossy()` for details.
///
/// # Panics
///
/// Panics if `n` has any interior NULs.
///
/// # Example
///
/// ```rust
/// use std::os;
///
/// let key = "HOME";
/// match os::getenv(key) {
///     Some(val) => println!("{}: {}", key, val),
///     None => println!("{} is not defined in the environment.", key)
/// }
/// ```
pub fn getenv(n: &str) -> Option<String> { unimplemented!() }

#[cfg(unix)]
/// Fetches the environment variable `n` byte vector from the current process,
/// returning None if the variable isn't set.
///
/// # Panics
///
/// Panics if `n` has any interior NULs.
pub fn getenv_as_bytes(n: &str) -> Option<Vec<u8>> { unimplemented!() }

#[cfg(windows)]
/// Fetches the environment variable `n` from the current process, returning
/// None if the variable isn't set.
pub fn getenv(n: &str) -> Option<String> { unimplemented!() }

#[cfg(windows)]
/// Fetches the environment variable `n` byte vector from the current process,
/// returning None if the variable isn't set.
pub fn getenv_as_bytes(n: &str) -> Option<Vec<u8>> { unimplemented!() }

/// Sets the environment variable `n` to the value `v` for the currently running
/// process.
///
/// # Example
///
/// ```rust
/// use std::os;
///
/// let key = "KEY";
/// os::setenv(key, "VALUE");
/// match os::getenv(key) {
///     Some(ref val) => println!("{}: {}", key, val),
///     None => println!("{} is not defined in the environment.", key)
/// }
/// ```
pub fn setenv<T: BytesContainer>(n: &str, v: T) { unimplemented!() }

/// Remove a variable from the environment entirely.
pub fn unsetenv(n: &str) { unimplemented!() }

/// Parses input according to platform conventions for the `PATH`
/// environment variable.
///
/// # Example
/// ```rust
/// use std::os;
///
/// let key = "PATH";
/// match os::getenv_as_bytes(key) {
///     Some(paths) => {
///         for path in os::split_paths(paths).iter() {
///             println!("'{}'", path.display());
///         }
///     }
///     None => println!("{} is not defined in the environment.", key)
/// }
/// ```
pub fn split_paths<T: BytesContainer>(unparsed: T) -> Vec<Path> { unimplemented!() }

/// A low-level OS in-memory pipe.
pub struct Pipe {
    /// A file descriptor representing the reading end of the pipe. Data written
    /// on the `out` file descriptor can be read from this file descriptor.
    pub reader: c_int,
    /// A file descriptor representing the write end of the pipe. Data written
    /// to this file descriptor can be read from the `input` file descriptor.
    pub writer: c_int,
}

/// Creates a new low-level OS in-memory pipe.
///
/// This function can fail to succeed if there are no more resources available
/// to allocate a pipe.
///
/// This function is also unsafe as there is no destructor associated with the
/// `Pipe` structure will return. If it is not arranged for the returned file
/// descriptors to be closed, the file descriptors will leak. For safe handling
/// of this scenario, use `std::io::PipeStream` instead.
pub unsafe fn pipe() -> IoResult<Pipe> { unimplemented!() }

/// Returns the proper dll filename for the given basename of a file
/// as a String.
#[cfg(not(target_os="ios"))]
pub fn dll_filename(base: &str) -> String { unimplemented!() }

/// Optionally returns the filesystem path to the current executable which is
/// running but with the executable name.
///
/// # Examples
///
/// ```rust
/// use std::os;
///
/// match os::self_exe_name() {
///     Some(exe_path) => println!("Path of this executable is: {}", exe_path.display()),
///     None => println!("Unable to get the path of this executable!")
/// };
/// ```
pub fn self_exe_name() -> Option<Path> { unimplemented!() }

/// Optionally returns the filesystem path to the current executable which is
/// running.
///
/// Like self_exe_name() but without the binary's name.
///
/// # Example
///
/// ```rust
/// use std::os;
///
/// match os::self_exe_path() {
///     Some(exe_path) => println!("Executable's Path is: {}", exe_path.display()),
///     None => println!("Impossible to fetch the path of this executable.")
/// };
/// ```
pub fn self_exe_path() -> Option<Path> { unimplemented!() }

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
/// use std::os;
///
/// match os::homedir() {
///     Some(ref p) => println!("{}", p.display()),
///     None => println!("Impossible to get your home dir!")
/// }
/// ```
pub fn homedir() -> Option<Path> { unimplemented!() }

/**
 * Returns the path to a temporary directory.
 *
 * On Unix, returns the value of the 'TMPDIR' environment variable if it is
 * set, otherwise for non-Android it returns '/tmp'. If Android, since there
 * is no global temporary folder (it is usually allocated per-app), we return
 * '/data/local/tmp'.
 *
 * On Windows, returns the value of, in order, the 'TMP', 'TEMP',
 * 'USERPROFILE' environment variable  if any are set and not the empty
 * string. Otherwise, tmpdir returns the path to the Windows directory.
 */
pub fn tmpdir() -> Path { unimplemented!() }

///
/// Convert a relative path to an absolute path
///
/// If the given path is relative, return it prepended with the current working
/// directory. If the given path is already an absolute path, return it
/// as is.
///
/// # Example
/// ```rust
/// use std::os;
/// use std::path::Path;
///
/// // Assume we're in a path like /home/someuser
/// let rel_path = Path::new("..");
/// let abs_path = os::make_absolute(&rel_path).unwrap();
/// println!("The absolute path is {}", abs_path.display());
/// // Prints "The absolute path is /home"
/// ```
// NB: this is here rather than in path because it is a form of environment
// querying; what it does depends on the process working directory, not just
// the input paths.
pub fn make_absolute(p: &Path) -> IoResult<Path> { unimplemented!() }

/// Changes the current working directory to the specified path, returning
/// whether the change was completed successfully or not.
///
/// # Example
/// ```rust
/// use std::os;
/// use std::path::Path;
///
/// let root = Path::new("/");
/// assert!(os::change_dir(&root).is_ok());
/// println!("Successfully changed working directory to {}!", root.display());
/// ```
pub fn change_dir(p: &Path) -> IoResult<()> { unimplemented!() }

/// Returns the platform-specific value of errno
pub fn errno() -> uint { unimplemented!() }

/// Return the string corresponding to an `errno()` value of `errnum`.
///
/// # Example
/// ```rust
/// use std::os;
///
/// // Same as println!("{}", last_os_error());
/// println!("{}", os::error_string(os::errno() as uint));
/// ```
pub fn error_string(errnum: uint) -> String { unimplemented!() }

/// Get a string representing the platform-dependent last error
pub fn last_os_error() -> String { unimplemented!() }

static EXIT_STATUS: AtomicInt = INIT_ATOMIC_INT;

/**
 * Sets the process exit code
 *
 * Sets the exit code returned by the process if all supervised tasks
 * terminate successfully (without panicking). If the current root task panics
 * and is supervised by the scheduler then any user-specified exit status is
 * ignored and the process exits with the default panic status.
 *
 * Note that this is not synchronized against modifications of other threads.
 */
pub fn set_exit_status(code: int) { unimplemented!() }

/// Fetches the process's current exit code. This defaults to 0 and can change
/// by calling `set_exit_status`.
pub fn get_exit_status() -> int { unimplemented!() }

#[cfg(target_os = "macos")]
unsafe fn load_argc_and_argv(argc: int,
                             argv: *const *const c_char) -> Vec<Vec<u8>> { unimplemented!() }

/**
 * Returns the command line arguments
 *
 * Returns a list of the command line arguments.
 */
#[cfg(target_os = "macos")]
fn real_args_as_bytes() -> Vec<Vec<u8>> { unimplemented!() }

// As _NSGetArgc and _NSGetArgv aren't mentioned in iOS docs
// and use underscores in their names - they're most probably
// are considered private and therefore should be avoided
// Here is another way to get arguments using Objective C
// runtime
//
// In general it looks like:
// res = Vec::new()
// let args = [[NSProcessInfo processInfo] arguments]
// for i in range(0, [args count])
//      res.push([args objectAtIndex:i])
// res
#[cfg(target_os = "ios")]
fn real_args_as_bytes() -> Vec<Vec<u8>> { unimplemented!() }

#[cfg(any(target_os = "linux",
          target_os = "android",
          target_os = "freebsd",
          target_os = "dragonfly"))]
fn real_args_as_bytes() -> Vec<Vec<u8>> { unimplemented!() }

#[cfg(not(windows))]
fn real_args() -> Vec<String> { unimplemented!() }

#[cfg(windows)]
fn real_args() -> Vec<String> { unimplemented!() }

#[cfg(windows)]
fn real_args_as_bytes() -> Vec<Vec<u8>> { unimplemented!() }

type LPCWSTR = *const u16;

#[cfg(windows)]
#[link_name="kernel32"]
extern "system" {
    fn GetCommandLineW() -> LPCWSTR;
    fn LocalFree(ptr: *mut c_void);
}

#[cfg(windows)]
#[link_name="shell32"]
extern "system" {
    fn CommandLineToArgvW(lpCmdLine: LPCWSTR,
                          pNumArgs: *mut c_int) -> *mut *mut u16;
}

/// Returns the arguments which this program was started with (normally passed
/// via the command line).
///
/// The first element is traditionally the path to the executable, but it can be
/// set to arbitrary text, and it may not even exist, so this property should not
/// be relied upon for security purposes.
///
/// The arguments are interpreted as utf-8, with invalid bytes replaced with \uFFFD.
/// See `String::from_utf8_lossy` for details.
/// # Example
///
/// ```rust
/// use std::os;
///
/// // Prints each argument on a separate line
/// for argument in os::args().iter() {
///     println!("{}", argument);
/// }
/// ```
pub fn args() -> Vec<String> { unimplemented!() }

/// Returns the arguments which this program was started with (normally passed
/// via the command line) as byte vectors.
pub fn args_as_bytes() -> Vec<Vec<u8>> { unimplemented!() }

#[cfg(target_os = "macos")]
extern {
    // These functions are in crt_externs.h.
    pub fn _NSGetArgc() -> *mut c_int;
    pub fn _NSGetArgv() -> *mut *mut *mut c_char;
}

// Round up `from` to be divisible by `to`
fn round_up(from: uint, to: uint) -> uint { unimplemented!() }

/// Returns the page size of the current architecture in bytes.
#[cfg(unix)]
pub fn page_size() -> uint { unimplemented!() }

/// Returns the page size of the current architecture in bytes.
#[cfg(windows)]
pub fn page_size() -> uint { unimplemented!() }

/// A memory mapped file or chunk of memory. This is a very system-specific
/// interface to the OS's memory mapping facilities (`mmap` on POSIX,
/// `VirtualAlloc`/`CreateFileMapping` on Windows). It makes no attempt at
/// abstracting platform differences, besides in error values returned. Consider
/// yourself warned.
///
/// The memory map is released (unmapped) when the destructor is run, so don't
/// let it leave scope by accident if you want it to stick around.
pub struct MemoryMap {
    data: *mut u8,
    len: uint,
    kind: MemoryMapKind,
}

/// Type of memory map
pub enum MemoryMapKind {
    /// Virtual memory map. Usually used to change the permissions of a given
    /// chunk of memory.  Corresponds to `VirtualAlloc` on Windows.
    MapFile(*const u8),
    /// Virtual memory map. Usually used to change the permissions of a given
    /// chunk of memory, or for allocation. Corresponds to `VirtualAlloc` on
    /// Windows.
    MapVirtual
}

/// Options the memory map is created with
pub enum MapOption {
    /// The memory should be readable
    MapReadable,
    /// The memory should be writable
    MapWritable,
    /// The memory should be executable
    MapExecutable,
    /// Create a map for a specific address range. Corresponds to `MAP_FIXED` on
    /// POSIX.
    MapAddr(*const u8),
    /// Create a memory mapping for a file with a given fd.
    MapFd(c_int),
    /// When using `MapFd`, the start of the map is `uint` bytes from the start
    /// of the file.
    MapOffset(uint),
    /// On POSIX, this can be used to specify the default flags passed to
    /// `mmap`. By default it uses `MAP_PRIVATE` and, if not using `MapFd`,
    /// `MAP_ANON`. This will override both of those. This is platform-specific
    /// (the exact values used) and ignored on Windows.
    MapNonStandardFlags(c_int),
}

/// Possible errors when creating a map.
pub enum MapError {
    /// ## The following are POSIX-specific
    ///
    /// fd was not open for reading or, if using `MapWritable`, was not open for
    /// writing.
    ErrFdNotAvail,
    /// fd was not valid
    ErrInvalidFd,
    /// Either the address given by `MapAddr` or offset given by `MapOffset` was
    /// not a multiple of `MemoryMap::granularity` (unaligned to page size).
    ErrUnaligned,
    /// With `MapFd`, the fd does not support mapping.
    ErrNoMapSupport,
    /// If using `MapAddr`, the address + `min_len` was outside of the process's
    /// address space. If using `MapFd`, the target of the fd didn't have enough
    /// resources to fulfill the request.
    ErrNoMem,
    /// A zero-length map was requested. This is invalid according to
    /// [POSIX](http://pubs.opengroup.org/onlinepubs/9699919799/functions/mmap.html).
    /// Not all platforms obey this, but this wrapper does.
    ErrZeroLength,
    /// Unrecognized error. The inner value is the unrecognized errno.
    ErrUnknown(int),
    /// ## The following are Windows-specific
    ///
    /// Unsupported combination of protection flags
    /// (`MapReadable`/`MapWritable`/`MapExecutable`).
    ErrUnsupProt,
    /// When using `MapFd`, `MapOffset` was given (Windows does not support this
    /// at all)
    ErrUnsupOffset,
    /// When using `MapFd`, there was already a mapping to the file.
    ErrAlreadyExists,
    /// Unrecognized error from `VirtualAlloc`. The inner value is the return
    /// value of GetLastError.
    ErrVirtualAlloc(uint),
    /// Unrecognized error from `CreateFileMapping`. The inner value is the
    /// return value of `GetLastError`.
    ErrCreateFileMappingW(uint),
    /// Unrecognized error from `MapViewOfFile`. The inner value is the return
    /// value of `GetLastError`.
    ErrMapViewOfFile(uint)
}

impl fmt::Show for MapError {
    fn fmt(&self, out: &mut fmt::Formatter) -> fmt::Result { unimplemented!() }
}

impl Error for MapError {
    fn description(&self) -> &str { unimplemented!() }
    fn detail(&self) -> Option<String> { unimplemented!() }
}

impl FromError<MapError> for Box<Error> {
    fn from_error(err: MapError) -> Box<Error> { unimplemented!() }
}

#[cfg(unix)]
impl MemoryMap {
    /// Create a new mapping with the given `options`, at least `min_len` bytes
    /// long. `min_len` must be greater than zero; see the note on
    /// `ErrZeroLength`.
    pub fn new(min_len: uint, options: &[MapOption]) -> Result<MemoryMap, MapError> { unimplemented!() }

    /// Granularity that the offset or address must be for `MapOffset` and
    /// `MapAddr` respectively.
    pub fn granularity() -> uint { unimplemented!() }
}

#[cfg(unix)]
impl Drop for MemoryMap {
    /// Unmap the mapping. Panics the task if `munmap` panics.
    fn drop(&mut self) { unimplemented!() }
}

#[cfg(windows)]
impl MemoryMap {
    /// Create a new mapping with the given `options`, at least `min_len` bytes long.
    pub fn new(min_len: uint, options: &[MapOption]) -> Result<MemoryMap, MapError> { unimplemented!() }

    /// Granularity of MapAddr() and MapOffset() parameter values.
    /// This may be greater than the value returned by page_size().
    pub fn granularity() -> uint { unimplemented!() }
}

#[cfg(windows)]
impl Drop for MemoryMap {
    /// Unmap the mapping. Panics the task if any of `VirtualFree`,
    /// `UnmapViewOfFile`, or `CloseHandle` fail.
    fn drop(&mut self) { unimplemented!() }
}

impl MemoryMap {
    /// Returns the pointer to the memory created or modified by this map.
    pub fn data(&self) -> *mut u8 { unimplemented!() }
    /// Returns the number of bytes this map applies to.
    pub fn len(&self) -> uint { unimplemented!() }
    /// Returns the type of mapping this represents.
    pub fn kind(&self) -> MemoryMapKind { unimplemented!() }
}

#[cfg(target_os = "linux")]
pub mod consts {
    pub use os::arch_consts::ARCH;

    pub const FAMILY: &'static str = "unix";

    /// A string describing the specific operating system in use: in this
    /// case, `linux`.
    pub const SYSNAME: &'static str = "linux";

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

#[cfg(target_os = "macos")]
pub mod consts {
    pub use os::arch_consts::ARCH;

    pub const FAMILY: &'static str = "unix";

    /// A string describing the specific operating system in use: in this
    /// case, `macos`.
    pub const SYSNAME: &'static str = "macos";

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

#[cfg(target_os = "ios")]
pub mod consts {
    pub use os::arch_consts::ARCH;

    pub const FAMILY: &'static str = "unix";

    /// A string describing the specific operating system in use: in this
    /// case, `ios`.
    pub const SYSNAME: &'static str = "ios";

    /// Specifies the filename suffix used for executable binaries on this
    /// platform: in this case, the empty string.
    pub const EXE_SUFFIX: &'static str = "";

    /// Specifies the file extension, if any, used for executable binaries
    /// on this platform: in this case, the empty string.
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "freebsd")]
pub mod consts {
    pub use os::arch_consts::ARCH;

    pub const FAMILY: &'static str = "unix";

    /// A string describing the specific operating system in use: in this
    /// case, `freebsd`.
    pub const SYSNAME: &'static str = "freebsd";

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

#[cfg(target_os = "dragonfly")]
pub mod consts {
    pub use os::arch_consts::ARCH;

    pub const FAMILY: &'static str = "unix";

    /// A string describing the specific operating system in use: in this
    /// case, `dragonfly`.
    pub const SYSNAME: &'static str = "dragonfly";

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

#[cfg(target_os = "android")]
pub mod consts {
    pub use os::arch_consts::ARCH;

    pub const FAMILY: &'static str = "unix";

    /// A string describing the specific operating system in use: in this
    /// case, `android`.
    pub const SYSNAME: &'static str = "android";

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

#[cfg(target_os = "windows")]
pub mod consts {
    pub use os::arch_consts::ARCH;

    pub const FAMILY: &'static str = "windows";

    /// A string describing the specific operating system in use: in this
    /// case, `windows`.
    pub const SYSNAME: &'static str = "windows";

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

#[cfg(target_arch = "mips")]
mod arch_consts {
    pub const ARCH: &'static str = "mips";
}

#[cfg(target_arch = "mipsel")]
mod arch_consts {
    pub const ARCH: &'static str = "mipsel";
}
