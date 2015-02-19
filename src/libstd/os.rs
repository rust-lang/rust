// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Higher-level interfaces to libc::* functions and operating system services.
//!
//! In general these take and return rust types, use rust idioms (enums,
//! closures, vectors) rather than C idioms, and do more extensive safety
//! checks.
//!
//! This module is not meant to only contain 1:1 mappings to libc entries; any
//! os-interface code that is reasonably useful and broadly applicable can go
//! here. Including utility routines that merely build on other os code.
//!
//! We assume the general case is that users do not care, and do not want to be
//! made to care, which operating system they are on. While they may want to
//! special case various special cases -- and so we will not _hide_ the facts of
//! which OS the user is on -- they should be given the opportunity to write
//! OS-ignorant code by default.

#![unstable(feature = "os")]

#![allow(missing_docs)]
#![allow(non_snake_case)]
#![allow(unused_imports)]

use self::MemoryMapKind::*;
use self::MapOption::*;
use self::MapError::*;

use boxed::Box;
use clone::Clone;
use env;
use error::{FromError, Error};
use ffi::{OsString, OsStr};
use fmt;
use iter::{Iterator, IteratorExt};
use libc::{c_void, c_int, c_char};
use libc;
use marker::{Copy, Send};
use old_io::{IoResult, IoError};
use ops::{Drop, FnOnce};
use option::Option::{Some, None};
use option::Option;
use old_path::{Path, GenericPath, BytesContainer};
use ptr::PtrExt;
use ptr;
use result::Result::{Err, Ok};
use result::Result;
use slice::{AsSlice, SliceExt};
use str::{Str, StrExt};
use str;
use string::{String, ToString};
use sync::atomic::{AtomicIsize, ATOMIC_ISIZE_INIT, Ordering};
use sys::os as os_imp;
use sys;
use vec::Vec;

#[cfg(unix)] use ffi::{self, CString};

#[cfg(unix)] pub use sys::ext as unix;
#[cfg(windows)] pub use sys::ext as windows;

/// Get the number of cores available
pub fn num_cpus() -> uint {
    unsafe {
        return rust_get_num_cpus() as uint;
    }

    extern {
        fn rust_get_num_cpus() -> libc::uintptr_t;
    }
}

pub const TMPBUF_SZ : uint = 1000;

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
/// // We assume that we are in a valid directory.
/// let current_working_directory = os::getcwd().unwrap();
/// println!("The current directory is {:?}", current_working_directory.display());
/// ```
#[deprecated(since = "1.0.0", reason = "renamed to std::env::current_dir")]
#[unstable(feature = "os")]
pub fn getcwd() -> IoResult<Path> {
    env::current_dir()
}

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
#[deprecated(since = "1.0.0", reason = "use env::vars instead")]
#[unstable(feature = "os")]
pub fn env() -> Vec<(String,String)> {
    env::vars_os().map(|(k, v)| {
        (k.to_string_lossy().into_owned(), v.to_string_lossy().into_owned())
    }).collect()
}

/// Returns a vector of (variable, value) byte-vector pairs for all the
/// environment variables of the current process.
#[deprecated(since = "1.0.0", reason = "use env::vars_os instead")]
#[unstable(feature = "os")]
pub fn env_as_bytes() -> Vec<(Vec<u8>, Vec<u8>)> {
    env::vars_os().map(|(k, v)| (byteify(k), byteify(v))).collect()
}

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
#[deprecated(since = "1.0.0", reason = "use env::var instead")]
#[unstable(feature = "os")]
pub fn getenv(n: &str) -> Option<String> {
    env::var(n).ok()
}

/// Fetches the environment variable `n` byte vector from the current process,
/// returning None if the variable isn't set.
///
/// # Panics
///
/// Panics if `n` has any interior NULs.
#[deprecated(since = "1.0.0", reason = "use env::var_os instead")]
#[unstable(feature = "os")]
pub fn getenv_as_bytes(n: &str) -> Option<Vec<u8>> {
    env::var_os(n).map(byteify)
}

#[cfg(unix)]
fn byteify(s: OsString) -> Vec<u8> {
    use os::unix::*;
    s.into_vec()
}
#[cfg(windows)]
fn byteify(s: OsString) -> Vec<u8> {
    s.to_string_lossy().as_bytes().to_vec()
}

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
#[deprecated(since = "1.0.0", reason = "renamed to env::set_var")]
#[unstable(feature = "os")]
pub fn setenv<T: BytesContainer>(n: &str, v: T) {
    #[cfg(unix)]
    fn _setenv(n: &str, v: &[u8]) {
        use os::unix::*;
        let v: OsString = OsStringExt::from_vec(v.to_vec());
        env::set_var(n, &v)
    }

    #[cfg(windows)]
    fn _setenv(n: &str, v: &[u8]) {
        let v = str::from_utf8(v).unwrap();
        env::set_var(n, v)
    }

    _setenv(n, v.container_as_bytes())
}

/// Remove a variable from the environment entirely.
#[deprecated(since = "1.0.0", reason = "renamed to env::remove_var")]
#[unstable(feature = "os")]
pub fn unsetenv(n: &str) {
    env::remove_var(n)
}

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
#[deprecated(since = "1.0.0", reason = "renamed to env::split_paths")]
#[unstable(feature = "os")]
pub fn split_paths<T: BytesContainer>(unparsed: T) -> Vec<Path> {
    let b = unparsed.container_as_bytes();
    let s = str::from_utf8(b).unwrap();
    env::split_paths(s).collect()
}

/// Joins a collection of `Path`s appropriately for the `PATH`
/// environment variable.
///
/// Returns a `Vec<u8>` on success, since `Path`s are not utf-8
/// encoded on all platforms.
///
/// Returns an `Err` (containing an error message) if one of the input
/// `Path`s contains an invalid character for constructing the `PATH`
/// variable (a double quote on Windows or a colon on Unix).
///
/// # Example
///
/// ```rust
/// use std::os;
/// use std::old_path::Path;
///
/// let key = "PATH";
/// let mut paths = os::getenv_as_bytes(key).map_or(Vec::new(), os::split_paths);
/// paths.push(Path::new("/home/xyz/bin"));
/// os::setenv(key, os::join_paths(paths.as_slice()).unwrap());
/// ```
#[deprecated(since = "1.0.0", reason = "renamed to env::join_paths")]
#[unstable(feature = "os")]
pub fn join_paths<T: BytesContainer>(paths: &[T]) -> Result<Vec<u8>, &'static str> {
    env::join_paths(paths.iter().map(|s| {
        str::from_utf8(s.container_as_bytes()).unwrap()
    })).map(|s| {
        s.to_string_lossy().into_owned().into_bytes()
    }).map_err(|_| "failed to join paths")
}

/// A low-level OS in-memory pipe.
#[derive(Copy)]
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
/// of this scenario, use `std::old_io::PipeStream` instead.
pub unsafe fn pipe() -> IoResult<Pipe> {
    let (reader, writer) = try!(sys::os::pipe());
    Ok(Pipe {
        reader: reader.unwrap(),
        writer: writer.unwrap(),
    })
}

/// Returns the proper dll filename for the given basename of a file
/// as a String.
#[cfg(not(target_os="ios"))]
#[deprecated(since = "1.0.0", reason = "this function will be removed, use the constants directly")]
#[unstable(feature = "os")]
#[allow(deprecated)]
pub fn dll_filename(base: &str) -> String {
    format!("{}{}{}", consts::DLL_PREFIX, base, consts::DLL_SUFFIX)
}

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
#[deprecated(since = "1.0.0", reason = "renamed to env::current_exe")]
#[unstable(feature = "os")]
pub fn self_exe_name() -> Option<Path> {
    env::current_exe().ok()
}

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
#[deprecated(since = "1.0.0", reason = "use env::current_exe + dir_path/pop")]
#[unstable(feature = "os")]
pub fn self_exe_path() -> Option<Path> {
    env::current_exe().ok().map(|mut p| { p.pop(); p })
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
/// use std::os;
///
/// match os::homedir() {
///     Some(ref p) => println!("{}", p.display()),
///     None => println!("Impossible to get your home dir!")
/// }
/// ```
#[deprecated(since = "1.0.0", reason = "renamed to env::home_dir")]
#[allow(deprecated)]
#[unstable(feature = "os")]
pub fn homedir() -> Option<Path> {
    #[inline]
    #[cfg(unix)]
    fn _homedir() -> Option<Path> {
        aux_homedir("HOME")
    }

    #[inline]
    #[cfg(windows)]
    fn _homedir() -> Option<Path> {
        aux_homedir("HOME").or(aux_homedir("USERPROFILE"))
    }

    #[inline]
    fn aux_homedir(home_name: &str) -> Option<Path> {
        match getenv_as_bytes(home_name) {
            Some(p)  => {
                if p.is_empty() { None } else { Path::new_opt(p) }
            },
            _ => None
        }
    }
    _homedir()
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
#[deprecated(since = "1.0.0", reason = "renamed to env::temp_dir")]
#[allow(deprecated)]
#[unstable(feature = "os")]
pub fn tmpdir() -> Path {
    return lookup();

    fn getenv_nonempty(v: &str) -> Option<Path> {
        match getenv(v) {
            Some(x) =>
                if x.is_empty() {
                    None
                } else {
                    Path::new_opt(x)
                },
            _ => None
        }
    }

    #[cfg(unix)]
    fn lookup() -> Path {
        let default = if cfg!(target_os = "android") {
            Path::new("/data/local/tmp")
        } else {
            Path::new("/tmp")
        };

        getenv_nonempty("TMPDIR").unwrap_or(default)
    }

    #[cfg(windows)]
    fn lookup() -> Path {
        getenv_nonempty("TMP").or(
            getenv_nonempty("TEMP").or(
                getenv_nonempty("USERPROFILE").or(
                   getenv_nonempty("WINDIR")))).unwrap_or(Path::new("C:\\Windows"))
    }
}

/// Convert a relative path to an absolute path
///
/// If the given path is relative, return it prepended with the current working
/// directory. If the given path is already an absolute path, return it
/// as is.
///
/// # Example
/// ```rust
/// use std::os;
/// use std::old_path::Path;
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
#[deprecated(since = "1.0.0", reason = "use env::current_dir + .join directly")]
#[unstable(feature = "os")]
pub fn make_absolute(p: &Path) -> IoResult<Path> {
    if p.is_absolute() {
        Ok(p.clone())
    } else {
        env::current_dir().map(|mut cwd| {
            cwd.push(p);
            cwd
        })
    }
}

/// Changes the current working directory to the specified path, returning
/// whether the change was completed successfully or not.
///
/// # Example
/// ```rust
/// use std::os;
/// use std::old_path::Path;
///
/// let root = Path::new("/");
/// assert!(os::change_dir(&root).is_ok());
/// println!("Successfully changed working directory to {}!", root.display());
/// ```
#[deprecated(since = "1.0.0", reason = "renamed to env::set_current_dir")]
#[unstable(feature = "os")]
pub fn change_dir(p: &Path) -> IoResult<()> {
    return sys::os::chdir(p);
}

/// Returns the platform-specific value of errno
pub fn errno() -> i32 {
    sys::os::errno() as i32
}

/// Return the string corresponding to an `errno()` value of `errnum`.
///
/// # Example
/// ```rust
/// use std::os;
///
/// // Same as println!("{}", last_os_error());
/// println!("{}", os::error_string(os::errno() as i32));
/// ```
pub fn error_string(errnum: i32) -> String {
    return sys::os::error_string(errnum);
}

/// Get a string representing the platform-dependent last error
pub fn last_os_error() -> String {
    error_string(errno())
}

/// Sets the process exit code
///
/// Sets the exit code returned by the process if all supervised tasks
/// terminate successfully (without panicking). If the current root task panics
/// and is supervised by the scheduler then any user-specified exit status is
/// ignored and the process exits with the default panic status.
///
/// Note that this is not synchronized against modifications of other threads.
#[deprecated(since = "1.0.0", reason = "renamed to env::set_exit_status")]
#[unstable(feature = "os")]
pub fn set_exit_status(code: int) {
    env::set_exit_status(code as i32)
}

/// Fetches the process's current exit code. This defaults to 0 and can change
/// by calling `set_exit_status`.
#[deprecated(since = "1.0.0", reason = "renamed to env::get_exit_status")]
#[unstable(feature = "os")]
pub fn get_exit_status() -> int {
    env::get_exit_status() as isize
}

#[cfg(target_os = "macos")]
unsafe fn load_argc_and_argv(argc: int,
                             argv: *const *const c_char) -> Vec<Vec<u8>> {
    use ffi::CStr;
    use iter::range;

    (0..argc).map(|i| {
        CStr::from_ptr(*argv.offset(i)).to_bytes().to_vec()
    }).collect()
}

/// Returns the command line arguments
///
/// Returns a list of the command line arguments.
#[cfg(target_os = "macos")]
fn real_args_as_bytes() -> Vec<Vec<u8>> {
    unsafe {
        let (argc, argv) = (*_NSGetArgc() as int,
                            *_NSGetArgv() as *const *const c_char);
        load_argc_and_argv(argc, argv)
    }
}

// As _NSGetArgc and _NSGetArgv aren't mentioned in iOS docs
// and use underscores in their names - they're most probably
// are considered private and therefore should be avoided
// Here is another way to get arguments using Objective C
// runtime
//
// In general it looks like:
// res = Vec::new()
// let args = [[NSProcessInfo processInfo] arguments]
// for i in 0..[args count]
//      res.push([args objectAtIndex:i])
// res
#[cfg(target_os = "ios")]
fn real_args_as_bytes() -> Vec<Vec<u8>> {
    use ffi::c_str_to_bytes;
    use iter::range;
    use mem;

    #[link(name = "objc")]
    extern {
        fn sel_registerName(name: *const libc::c_uchar) -> Sel;
        fn objc_msgSend(obj: NsId, sel: Sel, ...) -> NsId;
        fn objc_getClass(class_name: *const libc::c_uchar) -> NsId;
    }

    #[link(name = "Foundation", kind = "framework")]
    extern {}

    type Sel = *const libc::c_void;
    type NsId = *const libc::c_void;

    let mut res = Vec::new();

    unsafe {
        let processInfoSel = sel_registerName("processInfo\0".as_ptr());
        let argumentsSel = sel_registerName("arguments\0".as_ptr());
        let utf8Sel = sel_registerName("UTF8String\0".as_ptr());
        let countSel = sel_registerName("count\0".as_ptr());
        let objectAtSel = sel_registerName("objectAtIndex:\0".as_ptr());

        let klass = objc_getClass("NSProcessInfo\0".as_ptr());
        let info = objc_msgSend(klass, processInfoSel);
        let args = objc_msgSend(info, argumentsSel);

        let cnt: int = mem::transmute(objc_msgSend(args, countSel));
        for i in 0..cnt {
            let tmp = objc_msgSend(args, objectAtSel, i);
            let utf_c_str: *const libc::c_char =
                mem::transmute(objc_msgSend(tmp, utf8Sel));
            res.push(c_str_to_bytes(&utf_c_str).to_vec());
        }
    }

    res
}

#[cfg(any(target_os = "linux",
          target_os = "android",
          target_os = "freebsd",
          target_os = "dragonfly",
          target_os = "openbsd"))]
fn real_args_as_bytes() -> Vec<Vec<u8>> {
    use rt;
    rt::args::clone().unwrap_or_else(|| vec![])
}

#[cfg(not(windows))]
fn real_args() -> Vec<String> {
    real_args_as_bytes().into_iter()
                        .map(|v| {
                            String::from_utf8_lossy(&v).into_owned()
                        }).collect()
}

#[cfg(windows)]
fn real_args() -> Vec<String> {
    use slice;
    use iter::range;

    let mut nArgs: c_int = 0;
    let lpArgCount: *mut c_int = &mut nArgs;
    let lpCmdLine = unsafe { GetCommandLineW() };
    let szArgList = unsafe { CommandLineToArgvW(lpCmdLine, lpArgCount) };

    let args: Vec<_> = (0..nArgs as uint).map(|i| unsafe {
        // Determine the length of this argument.
        let ptr = *szArgList.offset(i as int);
        let mut len = 0;
        while *ptr.offset(len as int) != 0 { len += 1; }

        // Push it onto the list.
        let ptr = ptr as *const u16;
        let buf = slice::from_raw_parts(ptr, len);
        let opt_s = String::from_utf16(sys::truncate_utf16_at_nul(buf));
        opt_s.ok().expect("CommandLineToArgvW returned invalid UTF-16")
    }).collect();

    unsafe {
        LocalFree(szArgList as *mut c_void);
    }

    return args
}

#[cfg(windows)]
fn real_args_as_bytes() -> Vec<Vec<u8>> {
    real_args().into_iter().map(|s| s.into_bytes()).collect()
}

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
#[deprecated(since = "1.0.0", reason = "use std::env::args() instead")]
#[unstable(feature = "os")]
pub fn args() -> Vec<String> {
    real_args()
}

/// Returns the arguments which this program was started with (normally passed
/// via the command line) as byte vectors.
#[deprecated(since = "1.0.0", reason = "use env::args_os instead")]
#[unstable(feature = "os")]
pub fn args_as_bytes() -> Vec<Vec<u8>> {
    real_args_as_bytes()
}

#[cfg(target_os = "macos")]
extern {
    // These functions are in crt_externs.h.
    fn _NSGetArgc() -> *mut c_int;
    fn _NSGetArgv() -> *mut *mut *mut c_char;
}

/// Returns the page size of the current architecture in bytes.
#[deprecated(since = "1.0.0", reason = "renamed to env::page_size")]
#[unstable(feature = "os")]
pub fn page_size() -> uint {
    sys::os::page_size()
}

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
#[allow(raw_pointer_derive)]
#[derive(Copy)]
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
#[allow(raw_pointer_derive)]
#[derive(Copy)]
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
    /// Create a memory mapping for a file with a given HANDLE.
    #[cfg(windows)]
    MapFd(libc::HANDLE),
    /// Create a memory mapping for a file with a given fd.
    #[cfg(not(windows))]
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
#[derive(Copy, Debug)]
pub enum MapError {
    /// # The following are POSIX-specific
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
    /// # The following are Windows-specific
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
    ErrVirtualAlloc(i32),
    /// Unrecognized error from `CreateFileMapping`. The inner value is the
    /// return value of `GetLastError`.
    ErrCreateFileMappingW(i32),
    /// Unrecognized error from `MapViewOfFile`. The inner value is the return
    /// value of `GetLastError`.
    ErrMapViewOfFile(i32)
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for MapError {
    fn fmt(&self, out: &mut fmt::Formatter) -> fmt::Result {
        let str = match *self {
            ErrFdNotAvail => "fd not available for reading or writing",
            ErrInvalidFd => "Invalid fd",
            ErrUnaligned => {
                "Unaligned address, invalid flags, negative length or \
                 unaligned offset"
            }
            ErrNoMapSupport=> "File doesn't support mapping",
            ErrNoMem => "Invalid address, or not enough available memory",
            ErrUnsupProt => "Protection mode unsupported",
            ErrUnsupOffset => "Offset in virtual memory mode is unsupported",
            ErrAlreadyExists => "File mapping for specified file already exists",
            ErrZeroLength => "Zero-length mapping not allowed",
            ErrUnknown(code) => {
                return write!(out, "Unknown error = {}", code)
            },
            ErrVirtualAlloc(code) => {
                return write!(out, "VirtualAlloc failure = {}", code)
            },
            ErrCreateFileMappingW(code) => {
                return write!(out, "CreateFileMappingW failure = {}", code)
            },
            ErrMapViewOfFile(code) => {
                return write!(out, "MapViewOfFile failure = {}", code)
            }
        };
        write!(out, "{}", str)
    }
}

impl Error for MapError {
    fn description(&self) -> &str { "memory map error" }
}

// Round up `from` to be divisible by `to`
fn round_up(from: uint, to: uint) -> uint {
    let r = if from % to == 0 {
        from
    } else {
        from + to - (from % to)
    };
    if r == 0 {
        to
    } else {
        r
    }
}

#[cfg(unix)]
impl MemoryMap {
    /// Create a new mapping with the given `options`, at least `min_len` bytes
    /// long. `min_len` must be greater than zero; see the note on
    /// `ErrZeroLength`.
    pub fn new(min_len: uint, options: &[MapOption]) -> Result<MemoryMap, MapError> {
        use libc::off_t;

        if min_len == 0 {
            return Err(ErrZeroLength)
        }
        let mut addr: *const u8 = ptr::null();
        let mut prot = 0;
        let mut flags = libc::MAP_PRIVATE;
        let mut fd = -1;
        let mut offset = 0;
        let mut custom_flags = false;
        let len = round_up(min_len, env::page_size());

        for &o in options {
            match o {
                MapReadable => { prot |= libc::PROT_READ; },
                MapWritable => { prot |= libc::PROT_WRITE; },
                MapExecutable => { prot |= libc::PROT_EXEC; },
                MapAddr(addr_) => {
                    flags |= libc::MAP_FIXED;
                    addr = addr_;
                },
                MapFd(fd_) => {
                    flags |= libc::MAP_FILE;
                    fd = fd_;
                },
                MapOffset(offset_) => { offset = offset_ as off_t; },
                MapNonStandardFlags(f) => { custom_flags = true; flags = f },
            }
        }
        if fd == -1 && !custom_flags { flags |= libc::MAP_ANON; }

        let r = unsafe {
            libc::mmap(addr as *mut c_void, len as libc::size_t, prot, flags,
                       fd, offset)
        };
        if r == libc::MAP_FAILED {
            Err(match errno() as c_int {
                libc::EACCES => ErrFdNotAvail,
                libc::EBADF => ErrInvalidFd,
                libc::EINVAL => ErrUnaligned,
                libc::ENODEV => ErrNoMapSupport,
                libc::ENOMEM => ErrNoMem,
                code => ErrUnknown(code as int)
            })
        } else {
            Ok(MemoryMap {
               data: r as *mut u8,
               len: len,
               kind: if fd == -1 {
                   MapVirtual
               } else {
                   MapFile(ptr::null())
               }
            })
        }
    }

    /// Granularity that the offset or address must be for `MapOffset` and
    /// `MapAddr` respectively.
    pub fn granularity() -> uint {
        env::page_size()
    }
}

#[cfg(unix)]
impl Drop for MemoryMap {
    /// Unmap the mapping. Panics the task if `munmap` panics.
    fn drop(&mut self) {
        if self.len == 0 { /* workaround for dummy_stack */ return; }

        unsafe {
            // `munmap` only panics due to logic errors
            libc::munmap(self.data as *mut c_void, self.len as libc::size_t);
        }
    }
}

#[cfg(windows)]
impl MemoryMap {
    /// Create a new mapping with the given `options`, at least `min_len` bytes long.
    pub fn new(min_len: uint, options: &[MapOption]) -> Result<MemoryMap, MapError> {
        use libc::types::os::arch::extra::{LPVOID, DWORD, SIZE_T, HANDLE};

        let mut lpAddress: LPVOID = ptr::null_mut();
        let mut readable = false;
        let mut writable = false;
        let mut executable = false;
        let mut handle: HANDLE = libc::INVALID_HANDLE_VALUE;
        let mut offset: uint = 0;
        let len = round_up(min_len, env::page_size());

        for &o in options {
            match o {
                MapReadable => { readable = true; },
                MapWritable => { writable = true; },
                MapExecutable => { executable = true; }
                MapAddr(addr_) => { lpAddress = addr_ as LPVOID; },
                MapFd(handle_) => { handle = handle_; },
                MapOffset(offset_) => { offset = offset_; },
                MapNonStandardFlags(..) => {}
            }
        }

        let flProtect = match (executable, readable, writable) {
            (false, false, false) if handle == libc::INVALID_HANDLE_VALUE => libc::PAGE_NOACCESS,
            (false, true, false) => libc::PAGE_READONLY,
            (false, true, true) => libc::PAGE_READWRITE,
            (true, false, false) if handle == libc::INVALID_HANDLE_VALUE => libc::PAGE_EXECUTE,
            (true, true, false) => libc::PAGE_EXECUTE_READ,
            (true, true, true) => libc::PAGE_EXECUTE_READWRITE,
            _ => return Err(ErrUnsupProt)
        };

        if handle == libc::INVALID_HANDLE_VALUE {
            if offset != 0 {
                return Err(ErrUnsupOffset);
            }
            let r = unsafe {
                libc::VirtualAlloc(lpAddress,
                                   len as SIZE_T,
                                   libc::MEM_COMMIT | libc::MEM_RESERVE,
                                   flProtect)
            };
            match r as uint {
                0 => Err(ErrVirtualAlloc(errno())),
                _ => Ok(MemoryMap {
                   data: r as *mut u8,
                   len: len,
                   kind: MapVirtual
                })
            }
        } else {
            let dwDesiredAccess = match (executable, readable, writable) {
                (false, true, false) => libc::FILE_MAP_READ,
                (false, true, true) => libc::FILE_MAP_WRITE,
                (true, true, false) => libc::FILE_MAP_READ | libc::FILE_MAP_EXECUTE,
                (true, true, true) => libc::FILE_MAP_WRITE | libc::FILE_MAP_EXECUTE,
                _ => return Err(ErrUnsupProt) // Actually, because of the check above,
                                              // we should never get here.
            };
            unsafe {
                let hFile = handle;
                let mapping = libc::CreateFileMappingW(hFile,
                                                       ptr::null_mut(),
                                                       flProtect,
                                                       0,
                                                       0,
                                                       ptr::null());
                if mapping == ptr::null_mut() {
                    return Err(ErrCreateFileMappingW(errno()));
                }
                if errno() as c_int == libc::ERROR_ALREADY_EXISTS {
                    return Err(ErrAlreadyExists);
                }
                let r = libc::MapViewOfFile(mapping,
                                            dwDesiredAccess,
                                            ((len as u64) >> 32) as DWORD,
                                            (offset & 0xffff_ffff) as DWORD,
                                            0);
                match r as uint {
                    0 => Err(ErrMapViewOfFile(errno())),
                    _ => Ok(MemoryMap {
                       data: r as *mut u8,
                       len: len,
                       kind: MapFile(mapping as *const u8)
                    })
                }
            }
        }
    }

    /// Granularity of MapAddr() and MapOffset() parameter values.
    /// This may be greater than the value returned by page_size().
    pub fn granularity() -> uint {
        use mem;
        unsafe {
            let mut info = mem::zeroed();
            libc::GetSystemInfo(&mut info);

            return info.dwAllocationGranularity as uint;
        }
    }
}

#[cfg(windows)]
impl Drop for MemoryMap {
    /// Unmap the mapping. Panics the task if any of `VirtualFree`,
    /// `UnmapViewOfFile`, or `CloseHandle` fail.
    fn drop(&mut self) {
        use libc::types::os::arch::extra::{LPCVOID, HANDLE};
        use libc::consts::os::extra::FALSE;
        if self.len == 0 { return }

        unsafe {
            match self.kind {
                MapVirtual => {
                    if libc::VirtualFree(self.data as *mut c_void, 0,
                                         libc::MEM_RELEASE) == 0 {
                        println!("VirtualFree failed: {}", errno());
                    }
                },
                MapFile(mapping) => {
                    if libc::UnmapViewOfFile(self.data as LPCVOID) == FALSE {
                        println!("UnmapViewOfFile failed: {}", errno());
                    }
                    if libc::CloseHandle(mapping as HANDLE) == FALSE {
                        println!("CloseHandle failed: {}", errno());
                    }
                }
            }
        }
    }
}

impl MemoryMap {
    /// Returns the pointer to the memory created or modified by this map.
    pub fn data(&self) -> *mut u8 { self.data }
    /// Returns the number of bytes this map applies to.
    pub fn len(&self) -> uint { self.len }
    /// Returns the type of mapping this represents.
    pub fn kind(&self) -> MemoryMapKind { self.kind }
}

#[cfg(target_os = "linux")]
#[deprecated(since = "1.0.0", reason = "renamed to env::consts")]
#[unstable(feature = "os")]
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
#[deprecated(since = "1.0.0", reason = "renamed to env::consts")]
#[unstable(feature = "os")]
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
#[deprecated(since = "1.0.0", reason = "renamed to env::consts")]
#[unstable(feature = "os")]
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
#[deprecated(since = "1.0.0", reason = "renamed to env::consts")]
#[unstable(feature = "os")]
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
#[deprecated(since = "1.0.0", reason = "renamed to env::consts")]
#[unstable(feature = "os")]
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

#[cfg(target_os = "openbsd")]
#[deprecated(since = "1.0.0", reason = "renamed to env::consts")]
#[unstable(feature = "os")]
pub mod consts {
    pub use os::arch_consts::ARCH;

    pub const FAMILY: &'static str = "unix";

    /// A string describing the specific operating system in use: in this
    /// case, `openbsd`.
    pub const SYSNAME: &'static str = "openbsd";

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
#[deprecated(since = "1.0.0", reason = "renamed to env::consts")]
#[unstable(feature = "os")]
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
#[deprecated(since = "1.0.0", reason = "renamed to env::consts")]
#[unstable(feature = "os")]
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
    #![allow(deprecated)] // rand

    use prelude::v1::*;

    use iter::repeat;
    use os::{env, getcwd, getenv, make_absolute};
    use os::{split_paths, join_paths, setenv, unsetenv};
    use os;
    use rand::Rng;
    use rand;

    #[test]
    pub fn last_os_error() {
        debug!("{}", os::last_os_error());
    }

    fn make_rand_name() -> String {
        let mut rng = rand::thread_rng();
        let n = format!("TEST{}", rng.gen_ascii_chars().take(10)
                                     .collect::<String>());
        assert!(getenv(&n).is_none());
        n
    }

    #[test]
    fn test_num_cpus() {
        assert!(os::num_cpus() > 0);
    }

    #[test]
    fn test_setenv() {
        let n = make_rand_name();
        setenv(&n, "VALUE");
        assert_eq!(getenv(&n), Some("VALUE".to_string()));
    }

    #[test]
    fn test_unsetenv() {
        let n = make_rand_name();
        setenv(&n, "VALUE");
        unsetenv(&n);
        assert_eq!(getenv(&n), None);
    }

    #[test]
    #[ignore]
    fn test_setenv_overwrite() {
        let n = make_rand_name();
        setenv(&n, "1");
        setenv(&n, "2");
        assert_eq!(getenv(&n), Some("2".to_string()));
        setenv(&n, "");
        assert_eq!(getenv(&n), Some("".to_string()));
    }

    // Windows GetEnvironmentVariable requires some extra work to make sure
    // the buffer the variable is copied into is the right size
    #[test]
    #[ignore]
    fn test_getenv_big() {
        let mut s = "".to_string();
        let mut i = 0;
        while i < 100 {
            s.push_str("aaaaaaaaaa");
            i += 1;
        }
        let n = make_rand_name();
        setenv(&n, &s);
        debug!("{}", s.clone());
        assert_eq!(getenv(&n), Some(s));
    }

    #[test]
    fn test_self_exe_name() {
        let path = os::self_exe_name();
        assert!(path.is_some());
        let path = path.unwrap();
        debug!("{}", path.display());

        // Hard to test this function
        assert!(path.is_absolute());
    }

    #[test]
    fn test_self_exe_path() {
        let path = os::self_exe_path();
        assert!(path.is_some());
        let path = path.unwrap();
        debug!("{}", path.display());

        // Hard to test this function
        assert!(path.is_absolute());
    }

    #[test]
    #[ignore]
    fn test_env_getenv() {
        let e = env();
        assert!(e.len() > 0);
        for p in &e {
            let (n, v) = (*p).clone();
            debug!("{}", n);
            let v2 = getenv(&n);
            // MingW seems to set some funky environment variables like
            // "=C:=C:\MinGW\msys\1.0\bin" and "!::=::\" that are returned
            // from env() but not visible from getenv().
            assert!(v2.is_none() || v2 == Some(v));
        }
    }

    #[test]
    fn test_env_set_get_huge() {
        let n = make_rand_name();
        let s = repeat("x").take(10000).collect::<String>();
        setenv(&n, &s);
        assert_eq!(getenv(&n), Some(s));
        unsetenv(&n);
        assert_eq!(getenv(&n), None);
    }

    #[test]
    fn test_env_setenv() {
        let n = make_rand_name();

        let mut e = env();
        setenv(&n, "VALUE");
        assert!(!e.contains(&(n.clone(), "VALUE".to_string())));

        e = env();
        assert!(e.contains(&(n, "VALUE".to_string())));
    }

    #[test]
    fn test() {
        assert!((!Path::new("test-path").is_absolute()));

        let cwd = getcwd().unwrap();
        debug!("Current working directory: {}", cwd.display());

        debug!("{}", make_absolute(&Path::new("test-path")).unwrap().display());
        debug!("{}", make_absolute(&Path::new("/usr/bin")).unwrap().display());
    }

    #[test]
    #[cfg(unix)]
    fn homedir() {
        let oldhome = getenv("HOME");

        setenv("HOME", "/home/MountainView");
        assert!(os::homedir() == Some(Path::new("/home/MountainView")));

        setenv("HOME", "");
        assert!(os::homedir().is_none());

        if let Some(s) = oldhome {
            setenv("HOME", s);
        }
    }

    #[test]
    #[cfg(windows)]
    fn homedir() {

        let oldhome = getenv("HOME");
        let olduserprofile = getenv("USERPROFILE");

        setenv("HOME", "");
        setenv("USERPROFILE", "");

        assert!(os::homedir().is_none());

        setenv("HOME", "/home/MountainView");
        assert!(os::homedir() == Some(Path::new("/home/MountainView")));

        setenv("HOME", "");

        setenv("USERPROFILE", "/home/MountainView");
        assert!(os::homedir() == Some(Path::new("/home/MountainView")));

        setenv("HOME", "/home/MountainView");
        setenv("USERPROFILE", "/home/PaloAlto");
        assert!(os::homedir() == Some(Path::new("/home/MountainView")));

        if let Some(s) = oldhome {
            setenv("HOME", &s);
        }
        if let Some(s) = olduserprofile {
            setenv("USERPROFILE", &s);
        }
    }

    #[test]
    fn memory_map_rw() {
        use result::Result::{Ok, Err};

        let chunk = match os::MemoryMap::new(16, &[
            os::MapOption::MapReadable,
            os::MapOption::MapWritable
        ]) {
            Ok(chunk) => chunk,
            Err(msg) => panic!("{:?}", msg)
        };
        assert!(chunk.len >= 16);

        unsafe {
            *chunk.data = 0xBE;
            assert!(*chunk.data == 0xBE);
        }
    }

    #[test]
    fn memory_map_file() {
        use libc;
        use os::*;
        use old_io::fs::{File, unlink};
        use old_io::SeekStyle::SeekSet;
        use old_io::FileMode::Open;
        use old_io::FileAccess::ReadWrite;

        #[cfg(not(windows))]
        fn get_fd(file: &File) -> libc::c_int {
            use os::unix::AsRawFd;
            file.as_raw_fd()
        }

        #[cfg(windows)]
        fn get_fd(file: &File) -> libc::HANDLE {
            use os::windows::AsRawHandle;
            file.as_raw_handle()
        }

        let mut path = tmpdir();
        path.push("mmap_file.tmp");
        let size = MemoryMap::granularity() * 2;
        let mut file = File::open_mode(&path, Open, ReadWrite).unwrap();
        file.seek(size as i64, SeekSet).unwrap();
        file.write_u8(0).unwrap();

        let chunk = MemoryMap::new(size / 2, &[
            MapOption::MapReadable,
            MapOption::MapWritable,
            MapOption::MapFd(get_fd(&file)),
            MapOption::MapOffset(size / 2)
        ]).unwrap();
        assert!(chunk.len > 0);

        unsafe {
            *chunk.data = 0xbe;
            assert!(*chunk.data == 0xbe);
        }
        drop(chunk);

        unlink(&path).unwrap();
    }

    #[test]
    #[cfg(windows)]
    fn split_paths_windows() {
        fn check_parse(unparsed: &str, parsed: &[&str]) -> bool {
            split_paths(unparsed) ==
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
            split_paths(unparsed) ==
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
            join_paths(input).unwrap() == output.as_bytes()
        }

        assert!(test_eq(&[], ""));
        assert!(test_eq(&["/bin", "/usr/bin", "/usr/local/bin"],
                         "/bin:/usr/bin:/usr/local/bin"));
        assert!(test_eq(&["", "/bin", "", "", "/usr/bin", ""],
                         ":/bin:::/usr/bin:"));
        assert!(join_paths(&["/te:st"]).is_err());
    }

    #[test]
    #[cfg(windows)]
    fn join_paths_windows() {
        fn test_eq(input: &[&str], output: &str) -> bool {
            join_paths(input).unwrap() == output.as_bytes()
        }

        assert!(test_eq(&[], ""));
        assert!(test_eq(&[r"c:\windows", r"c:\"],
                        r"c:\windows;c:\"));
        assert!(test_eq(&["", r"c:\windows", "", "", r"c:\", ""],
                        r";c:\windows;;;c:\;"));
        assert!(test_eq(&[r"c:\te;st", r"c:\"],
                        r#""c:\te;st";c:\"#));
        assert!(join_paths(&[r#"c:\te"st"#]).is_err());
    }

    // More recursive_mkdir tests are in extra::tempfile
}
