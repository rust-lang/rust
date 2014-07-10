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

#![allow(missing_doc)]
#![allow(non_snake_case_functions)]

use clone::Clone;
use collections::Collection;
use fmt;
use io::{IoResult, IoError};
use iter::Iterator;
use libc::{c_void, c_int};
use libc;
use ops::Drop;
use option::{Some, None, Option};
use os;
use path::{Path, GenericPath, BytesContainer};
use ptr::RawPtr;
use ptr;
use result::{Err, Ok, Result};
use slice::{Vector, ImmutableVector, MutableVector, ImmutableEqVector};
use str::{Str, StrSlice, StrAllocating};
use str;
use string::String;
use sync::atomics::{AtomicInt, INIT_ATOMIC_INT, SeqCst};
use vec::Vec;

#[cfg(unix)]
use c_str::ToCStr;
#[cfg(unix)]
use libc::c_char;
#[cfg(windows)]
use str::OwnedStr;

/// Get the number of cores available
pub fn num_cpus() -> uint {
    unsafe {
        return rust_get_num_cpus();
    }

    extern {
        fn rust_get_num_cpus() -> libc::uintptr_t;
    }
}

pub static TMPBUF_SZ : uint = 1000u;
static BUF_BYTES : uint = 2048u;

/// Returns the current working directory as a Path.
///
/// # Failure
///
/// Fails if the current working directory value is invalid:
/// Possibles cases:
///
/// * Current directory does not exist.
/// * There are insufficient permissions to access the current directory.
///
/// # Example
///
/// ```rust
/// use std::os;
///
/// // We assume that we are in a valid directory like "/home".
/// let current_working_directory = os::getcwd();
/// println!("The current directory is {}", current_working_directory.display());
/// // /home
/// ```
#[cfg(unix)]
pub fn getcwd() -> Path {
    use c_str::CString;

    let mut buf = [0 as c_char, ..BUF_BYTES];
    unsafe {
        if libc::getcwd(buf.as_mut_ptr(), buf.len() as libc::size_t).is_null() {
            fail!()
        }
        Path::new(CString::new(buf.as_ptr(), false))
    }
}

/// Returns the current working directory as a Path.
///
/// # Failure
///
/// Fails if the current working directory value is invalid.
/// Possibles cases:
///
/// * Current directory does not exist.
/// * There are insufficient permissions to access the current directory.
///
/// # Example
///
/// ```rust
/// use std::os;
///
/// // We assume that we are in a valid directory like "C:\\Windows".
/// let current_working_directory = os::getcwd();
/// println!("The current directory is {}", current_working_directory.display());
/// // C:\\Windows
/// ```
#[cfg(windows)]
pub fn getcwd() -> Path {
    use libc::DWORD;
    use libc::GetCurrentDirectoryW;

    let mut buf = [0 as u16, ..BUF_BYTES];
    unsafe {
        if libc::GetCurrentDirectoryW(buf.len() as DWORD, buf.as_mut_ptr()) == 0 as DWORD {
            fail!();
        }
    }
    Path::new(str::from_utf16(str::truncate_utf16_at_nul(buf))
              .expect("GetCurrentDirectoryW returned invalid UTF-16"))
}

#[cfg(windows)]
pub mod win32 {
    use libc::types::os::arch::extra::DWORD;
    use libc;
    use option::{None, Option};
    use option;
    use os::TMPBUF_SZ;
    use slice::{MutableVector, ImmutableVector};
    use string::String;
    use str::StrSlice;
    use str;
    use vec::Vec;

    pub fn fill_utf16_buf_and_decode(f: |*mut u16, DWORD| -> DWORD)
        -> Option<String> {

        unsafe {
            let mut n = TMPBUF_SZ as DWORD;
            let mut res = None;
            let mut done = false;
            while !done {
                let mut buf = Vec::from_elem(n as uint, 0u16);
                let k = f(buf.as_mut_ptr(), n);
                if k == (0 as DWORD) {
                    done = true;
                } else if k == n &&
                          libc::GetLastError() ==
                          libc::ERROR_INSUFFICIENT_BUFFER as DWORD {
                    n *= 2 as DWORD;
                } else if k >= n {
                    n = k;
                } else {
                    done = true;
                }
                if k != 0 && done {
                    let sub = buf.slice(0, k as uint);
                    // We want to explicitly catch the case when the
                    // closure returned invalid UTF-16, rather than
                    // set `res` to None and continue.
                    let s = str::from_utf16(sub)
                        .expect("fill_utf16_buf_and_decode: closure created invalid UTF-16");
                    res = option::Some(s)
                }
            }
            return res;
        }
    }
}

/*
Accessing environment variables is not generally threadsafe.
Serialize access through a global lock.
*/
fn with_env_lock<T>(f: || -> T) -> T {
    use rt::mutex::{StaticNativeMutex, NATIVE_MUTEX_INIT};

    static mut lock: StaticNativeMutex = NATIVE_MUTEX_INIT;

    unsafe {
        let _guard = lock.lock();
        f()
    }
}

/// Returns a vector of (variable, value) pairs, for all the environment
/// variables of the current process.
///
/// Invalid UTF-8 bytes are replaced with \uFFFD. See `str::from_utf8_lossy()`
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
pub fn env() -> Vec<(String,String)> {
    env_as_bytes().move_iter().map(|(k,v)| {
        let k = String::from_str(str::from_utf8_lossy(k.as_slice()).as_slice());
        let v = String::from_str(str::from_utf8_lossy(v.as_slice()).as_slice());
        (k,v)
    }).collect()
}

/// Returns a vector of (variable, value) byte-vector pairs for all the
/// environment variables of the current process.
pub fn env_as_bytes() -> Vec<(Vec<u8>,Vec<u8>)> {
    unsafe {
        #[cfg(windows)]
        unsafe fn get_env_pairs() -> Vec<Vec<u8>> {
            use slice::raw;

            use libc::funcs::extra::kernel32::{
                GetEnvironmentStringsW,
                FreeEnvironmentStringsW
            };
            let ch = GetEnvironmentStringsW();
            if ch as uint == 0 {
                fail!("os::env() failure getting env string from OS: {}",
                       os::last_os_error());
            }
            // Here, we lossily decode the string as UTF16.
            //
            // The docs suggest that the result should be in Unicode, but
            // Windows doesn't guarantee it's actually UTF16 -- it doesn't
            // validate the environment string passed to CreateProcess nor
            // SetEnvironmentVariable.  Yet, it's unlikely that returning a
            // raw u16 buffer would be of practical use since the result would
            // be inherently platform-dependent and introduce additional
            // complexity to this code.
            //
            // Using the non-Unicode version of GetEnvironmentStrings is even
            // worse since the result is in an OEM code page.  Characters that
            // can't be encoded in the code page would be turned into question
            // marks.
            let mut result = Vec::new();
            let mut i = 0;
            while *ch.offset(i) != 0 {
                let p = &*ch.offset(i);
                let len = ptr::position(p, |c| *c == 0);
                raw::buf_as_slice(p, len, |s| {
                    result.push(str::from_utf16_lossy(s).into_bytes());
                });
                i += len as int + 1;
            }
            FreeEnvironmentStringsW(ch);
            result
        }
        #[cfg(unix)]
        unsafe fn get_env_pairs() -> Vec<Vec<u8>> {
            use c_str::CString;

            extern {
                fn rust_env_pairs() -> *const *const c_char;
            }
            let environ = rust_env_pairs();
            if environ as uint == 0 {
                fail!("os::env() failure getting env string from OS: {}",
                       os::last_os_error());
            }
            let mut result = Vec::new();
            ptr::array_each(environ, |e| {
                let env_pair =
                    Vec::from_slice(CString::new(e, false).as_bytes_no_nul());
                result.push(env_pair);
            });
            result
        }

        fn env_convert(input: Vec<Vec<u8>>) -> Vec<(Vec<u8>, Vec<u8>)> {
            let mut pairs = Vec::new();
            for p in input.iter() {
                let mut it = p.as_slice().splitn(1, |b| *b == '=' as u8);
                let key = Vec::from_slice(it.next().unwrap());
                let val = Vec::from_slice(it.next().unwrap_or(&[]));
                pairs.push((key, val));
            }
            pairs
        }
        with_env_lock(|| {
            let unparsed_environ = get_env_pairs();
            env_convert(unparsed_environ)
        })
    }
}

#[cfg(unix)]
/// Fetches the environment variable `n` from the current process, returning
/// None if the variable isn't set.
///
/// Any invalid UTF-8 bytes in the value are replaced by \uFFFD. See
/// `str::from_utf8_lossy()` for details.
///
/// # Failure
///
/// Fails if `n` has any interior NULs.
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
pub fn getenv(n: &str) -> Option<String> {
    getenv_as_bytes(n).map(|v| String::from_str(str::from_utf8_lossy(v.as_slice()).as_slice()))
}

#[cfg(unix)]
/// Fetches the environment variable `n` byte vector from the current process,
/// returning None if the variable isn't set.
///
/// # Failure
///
/// Fails if `n` has any interior NULs.
pub fn getenv_as_bytes(n: &str) -> Option<Vec<u8>> {
    use c_str::CString;

    unsafe {
        with_env_lock(|| {
            let s = n.with_c_str(|buf| libc::getenv(buf));
            if s.is_null() {
                None
            } else {
                Some(Vec::from_slice(CString::new(s as *const i8,
                                                  false).as_bytes_no_nul()))
            }
        })
    }
}

#[cfg(windows)]
/// Fetches the environment variable `n` from the current process, returning
/// None if the variable isn't set.
pub fn getenv(n: &str) -> Option<String> {
    unsafe {
        with_env_lock(|| {
            use os::win32::{fill_utf16_buf_and_decode};
            let n: Vec<u16> = n.utf16_units().collect();
            let n = n.append_one(0);
            fill_utf16_buf_and_decode(|buf, sz| {
                libc::GetEnvironmentVariableW(n.as_ptr(), buf, sz)
            })
        })
    }
}

#[cfg(windows)]
/// Fetches the environment variable `n` byte vector from the current process,
/// returning None if the variable isn't set.
pub fn getenv_as_bytes(n: &str) -> Option<Vec<u8>> {
    getenv(n).map(|s| s.into_bytes())
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
pub fn setenv<T: BytesContainer>(n: &str, v: T) {
    #[cfg(unix)]
    fn _setenv(n: &str, v: &[u8]) {
        unsafe {
            with_env_lock(|| {
                n.with_c_str(|nbuf| {
                    v.with_c_str(|vbuf| {
                        libc::funcs::posix01::unistd::setenv(nbuf, vbuf, 1);
                    })
                })
            })
        }
    }

    #[cfg(windows)]
    fn _setenv(n: &str, v: &[u8]) {
        let n: Vec<u16> = n.utf16_units().collect();
        let n = n.append_one(0);
        let v: Vec<u16> = str::from_utf8(v).unwrap().utf16_units().collect();
        let v = v.append_one(0);

        unsafe {
            with_env_lock(|| {
                libc::SetEnvironmentVariableW(n.as_ptr(), v.as_ptr());
            })
        }
    }

    _setenv(n, v.container_as_bytes())
}

/// Remove a variable from the environment entirely.
pub fn unsetenv(n: &str) {
    #[cfg(unix)]
    fn _unsetenv(n: &str) {
        unsafe {
            with_env_lock(|| {
                n.with_c_str(|nbuf| {
                    libc::funcs::posix01::unistd::unsetenv(nbuf);
                })
            })
        }
    }

    #[cfg(windows)]
    fn _unsetenv(n: &str) {
        let n: Vec<u16> = n.utf16_units().collect();
        let n = n.append_one(0);
        unsafe {
            with_env_lock(|| {
                libc::SetEnvironmentVariableW(n.as_ptr(), ptr::null());
            })
        }
    }
    _unsetenv(n);
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
pub fn split_paths<T: BytesContainer>(unparsed: T) -> Vec<Path> {
    #[cfg(unix)]
    fn _split_paths<T: BytesContainer>(unparsed: T) -> Vec<Path> {
        unparsed.container_as_bytes()
                .split(|b| *b == b':')
                .map(Path::new)
                .collect()
    }

    #[cfg(windows)]
    fn _split_paths<T: BytesContainer>(unparsed: T) -> Vec<Path> {
        // On Windows, the PATH environment variable is semicolon separated.  Double
        // quotes are used as a way of introducing literal semicolons (since
        // c:\some;dir is a valid Windows path). Double quotes are not themselves
        // permitted in path names, so there is no way to escape a double quote.
        // Quoted regions can appear in arbitrary locations, so
        //
        //   c:\foo;c:\som"e;di"r;c:\bar
        //
        // Should parse as [c:\foo, c:\some;dir, c:\bar].
        //
        // (The above is based on testing; there is no clear reference available
        // for the grammar.)

        let mut parsed = Vec::new();
        let mut in_progress = Vec::new();
        let mut in_quote = false;

        for b in unparsed.container_as_bytes().iter() {
            match *b {
                b';' if !in_quote => {
                    parsed.push(Path::new(in_progress.as_slice()));
                    in_progress.truncate(0)
                }
                b'"' => {
                    in_quote = !in_quote;
                }
                _  => {
                    in_progress.push(*b);
                }
            }
        }
        parsed.push(Path::new(in_progress));
        parsed
    }

    _split_paths(unparsed)
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
/// use std::path::Path;
///
/// let key = "PATH";
/// let mut paths = os::getenv_as_bytes(key).map_or(Vec::new(), os::split_paths);
/// paths.push(Path::new("/home/xyz/bin"));
/// os::setenv(key, os::join_paths(paths.as_slice()).unwrap());
/// ```
pub fn join_paths<T: BytesContainer>(paths: &[T]) -> Result<Vec<u8>, &'static str> {
    #[cfg(windows)]
    fn _join_paths<T: BytesContainer>(paths: &[T]) -> Result<Vec<u8>, &'static str> {
        let mut joined = Vec::new();
        let sep = b';';

        for (i, path) in paths.iter().map(|p| p.container_as_bytes()).enumerate() {
            if i > 0 { joined.push(sep) }
            if path.contains(&b'"') {
                return Err("path segment contains `\"`");
            } else if path.contains(&sep) {
                joined.push(b'"');
                joined.push_all(path);
                joined.push(b'"');
            } else {
                joined.push_all(path);
            }
        }

        Ok(joined)
    }

    #[cfg(unix)]
    fn _join_paths<T: BytesContainer>(paths: &[T]) -> Result<Vec<u8>, &'static str> {
        let mut joined = Vec::new();
        let sep = b':';

        for (i, path) in paths.iter().map(|p| p.container_as_bytes()).enumerate() {
            if i > 0 { joined.push(sep) }
            if path.contains(&sep) { return Err("path segment contains separator `:`") }
            joined.push_all(path);
        }

        Ok(joined)
    }

    _join_paths(paths)
}

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
pub unsafe fn pipe() -> IoResult<Pipe> {
    return _pipe();

    #[cfg(unix)]
    unsafe fn _pipe() -> IoResult<Pipe> {
        let mut fds = [0, ..2];
        match libc::pipe(fds.as_mut_ptr()) {
            0 => Ok(Pipe { reader: fds[0], writer: fds[1] }),
            _ => Err(IoError::last_error()),
        }
    }

    #[cfg(windows)]
    unsafe fn _pipe() -> IoResult<Pipe> {
        // Windows pipes work subtly differently than unix pipes, and their
        // inheritance has to be handled in a different way that I do not
        // fully understand. Here we explicitly make the pipe non-inheritable,
        // which means to pass it to a subprocess they need to be duplicated
        // first, as in std::run.
        let mut fds = [0, ..2];
        match libc::pipe(fds.as_mut_ptr(), 1024 as ::libc::c_uint,
                         (libc::O_BINARY | libc::O_NOINHERIT) as c_int) {
            0 => {
                assert!(fds[0] != -1 && fds[0] != 0);
                assert!(fds[1] != -1 && fds[1] != 0);
                Ok(Pipe { reader: fds[0], writer: fds[1] })
            }
            _ => Err(IoError::last_error()),
        }
    }
}

/// Returns the proper dll filename for the given basename of a file
/// as a String.
#[cfg(not(target_os="ios"))]
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
pub fn self_exe_name() -> Option<Path> {

    #[cfg(target_os = "freebsd")]
    fn load_self() -> Option<Vec<u8>> {
        unsafe {
            use libc::funcs::bsd44::*;
            use libc::consts::os::extra::*;
            let mut mib = vec![CTL_KERN as c_int,
                               KERN_PROC as c_int,
                               KERN_PROC_PATHNAME as c_int,
                               -1 as c_int];
            let mut sz: libc::size_t = 0;
            let err = sysctl(mib.as_mut_ptr(), mib.len() as ::libc::c_uint,
                             ptr::mut_null(), &mut sz, ptr::mut_null(),
                             0u as libc::size_t);
            if err != 0 { return None; }
            if sz == 0 { return None; }
            let mut v: Vec<u8> = Vec::with_capacity(sz as uint);
            let err = sysctl(mib.as_mut_ptr(), mib.len() as ::libc::c_uint,
                             v.as_mut_ptr() as *mut c_void, &mut sz,
                             ptr::mut_null(), 0u as libc::size_t);
            if err != 0 { return None; }
            if sz == 0 { return None; }
            v.set_len(sz as uint - 1); // chop off trailing NUL
            Some(v)
        }
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    fn load_self() -> Option<Vec<u8>> {
        use std::io;

        match io::fs::readlink(&Path::new("/proc/self/exe")) {
            Ok(path) => Some(path.into_vec()),
            Err(..) => None
        }
    }

    #[cfg(target_os = "macos")]
    #[cfg(target_os = "ios")]
    fn load_self() -> Option<Vec<u8>> {
        unsafe {
            use libc::funcs::extra::_NSGetExecutablePath;
            let mut sz: u32 = 0;
            _NSGetExecutablePath(ptr::mut_null(), &mut sz);
            if sz == 0 { return None; }
            let mut v: Vec<u8> = Vec::with_capacity(sz as uint);
            let err = _NSGetExecutablePath(v.as_mut_ptr() as *mut i8, &mut sz);
            if err != 0 { return None; }
            v.set_len(sz as uint - 1); // chop off trailing NUL
            Some(v)
        }
    }

    #[cfg(windows)]
    fn load_self() -> Option<Vec<u8>> {
        use str::OwnedStr;

        unsafe {
            use os::win32::fill_utf16_buf_and_decode;
            fill_utf16_buf_and_decode(|buf, sz| {
                libc::GetModuleFileNameW(0u as libc::DWORD, buf, sz)
            }).map(|s| s.into_string().into_bytes())
        }
    }

    load_self().and_then(Path::new_opt)
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
pub fn self_exe_path() -> Option<Path> {
    self_exe_name().map(|mut p| { p.pop(); p })
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

/**
 * Convert a relative path to an absolute path
 *
 * If the given path is relative, return it prepended with the current working
 * directory. If the given path is already an absolute path, return it
 * as is.
 */
// NB: this is here rather than in path because it is a form of environment
// querying; what it does depends on the process working directory, not just
// the input paths.
pub fn make_absolute(p: &Path) -> Path {
    if p.is_absolute() {
        p.clone()
    } else {
        let mut ret = getcwd();
        ret.push(p);
        ret
    }
}

/// Changes the current working directory to the specified path, returning
/// whether the change was completed successfully or not.
pub fn change_dir(p: &Path) -> bool {
    return chdir(p);

    #[cfg(windows)]
    fn chdir(p: &Path) -> bool {
        let p = match p.as_str() {
            Some(s) => s.utf16_units().collect::<Vec<u16>>().append_one(0),
            None => return false,
        };
        unsafe {
            libc::SetCurrentDirectoryW(p.as_ptr()) != (0 as libc::BOOL)
        }
    }

    #[cfg(unix)]
    fn chdir(p: &Path) -> bool {
        p.with_c_str(|buf| {
            unsafe {
                libc::chdir(buf) == (0 as c_int)
            }
        })
    }
}

#[cfg(unix)]
/// Returns the platform-specific value of errno
pub fn errno() -> int {
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "ios")]
    #[cfg(target_os = "freebsd")]
    fn errno_location() -> *const c_int {
        extern {
            fn __error() -> *const c_int;
        }
        unsafe {
            __error()
        }
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    fn errno_location() -> *const c_int {
        extern {
            fn __errno_location() -> *const c_int;
        }
        unsafe {
            __errno_location()
        }
    }

    unsafe {
        (*errno_location()) as int
    }
}

#[cfg(windows)]
/// Returns the platform-specific value of errno
pub fn errno() -> uint {
    use libc::types::os::arch::extra::DWORD;

    #[link_name = "kernel32"]
    extern "system" {
        fn GetLastError() -> DWORD;
    }

    unsafe {
        GetLastError() as uint
    }
}

/// Return the string corresponding to an `errno()` value of `errnum`.
pub fn error_string(errnum: uint) -> String {
    return strerror(errnum);

    #[cfg(unix)]
    fn strerror(errnum: uint) -> String {
        #[cfg(target_os = "macos")]
        #[cfg(target_os = "ios")]
        #[cfg(target_os = "android")]
        #[cfg(target_os = "freebsd")]
        fn strerror_r(errnum: c_int, buf: *mut c_char, buflen: libc::size_t)
                      -> c_int {
            extern {
                fn strerror_r(errnum: c_int, buf: *mut c_char,
                              buflen: libc::size_t) -> c_int;
            }
            unsafe {
                strerror_r(errnum, buf, buflen)
            }
        }

        // GNU libc provides a non-compliant version of strerror_r by default
        // and requires macros to instead use the POSIX compliant variant.
        // So we just use __xpg_strerror_r which is always POSIX compliant
        #[cfg(target_os = "linux")]
        fn strerror_r(errnum: c_int, buf: *mut c_char,
                      buflen: libc::size_t) -> c_int {
            extern {
                fn __xpg_strerror_r(errnum: c_int,
                                    buf: *mut c_char,
                                    buflen: libc::size_t)
                                    -> c_int;
            }
            unsafe {
                __xpg_strerror_r(errnum, buf, buflen)
            }
        }

        let mut buf = [0 as c_char, ..TMPBUF_SZ];

        let p = buf.as_mut_ptr();
        unsafe {
            if strerror_r(errnum as c_int, p, buf.len() as libc::size_t) < 0 {
                fail!("strerror_r failure");
            }

            str::raw::from_c_str(p as *const c_char).into_string()
        }
    }

    #[cfg(windows)]
    fn strerror(errnum: uint) -> String {
        use libc::types::os::arch::extra::DWORD;
        use libc::types::os::arch::extra::LPWSTR;
        use libc::types::os::arch::extra::LPVOID;
        use libc::types::os::arch::extra::WCHAR;

        #[link_name = "kernel32"]
        extern "system" {
            fn FormatMessageW(flags: DWORD,
                              lpSrc: LPVOID,
                              msgId: DWORD,
                              langId: DWORD,
                              buf: LPWSTR,
                              nsize: DWORD,
                              args: *const c_void)
                              -> DWORD;
        }

        static FORMAT_MESSAGE_FROM_SYSTEM: DWORD = 0x00001000;
        static FORMAT_MESSAGE_IGNORE_INSERTS: DWORD = 0x00000200;

        // This value is calculated from the macro
        // MAKELANGID(LANG_SYSTEM_DEFAULT, SUBLANG_SYS_DEFAULT)
        let langId = 0x0800 as DWORD;

        let mut buf = [0 as WCHAR, ..TMPBUF_SZ];

        unsafe {
            let res = FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM |
                                     FORMAT_MESSAGE_IGNORE_INSERTS,
                                     ptr::mut_null(),
                                     errnum as DWORD,
                                     langId,
                                     buf.as_mut_ptr(),
                                     buf.len() as DWORD,
                                     ptr::null());
            if res == 0 {
                // Sometimes FormatMessageW can fail e.g. system doesn't like langId,
                let fm_err = errno();
                return format!("OS Error {} (FormatMessageW() returned error {})", errnum, fm_err);
            }

            let msg = str::from_utf16(str::truncate_utf16_at_nul(buf));
            match msg {
                Some(msg) => format!("OS Error {}: {}", errnum, msg),
                None => format!("OS Error {} (FormatMessageW() returned invalid UTF-16)", errnum),
            }
        }
    }
}

/// Get a string representing the platform-dependent last error
pub fn last_os_error() -> String {
    error_string(errno() as uint)
}

static mut EXIT_STATUS: AtomicInt = INIT_ATOMIC_INT;

/**
 * Sets the process exit code
 *
 * Sets the exit code returned by the process if all supervised tasks
 * terminate successfully (without failing). If the current root task fails
 * and is supervised by the scheduler then any user-specified exit status is
 * ignored and the process exits with the default failure status.
 *
 * Note that this is not synchronized against modifications of other threads.
 */
pub fn set_exit_status(code: int) {
    unsafe { EXIT_STATUS.store(code, SeqCst) }
}

/// Fetches the process's current exit code. This defaults to 0 and can change
/// by calling `set_exit_status`.
pub fn get_exit_status() -> int {
    unsafe { EXIT_STATUS.load(SeqCst) }
}

#[cfg(target_os = "macos")]
unsafe fn load_argc_and_argv(argc: int,
                             argv: *const *const c_char) -> Vec<Vec<u8>> {
    use c_str::CString;

    Vec::from_fn(argc as uint, |i| {
        Vec::from_slice(CString::new(*argv.offset(i as int),
                                     false).as_bytes_no_nul())
    })
}

/**
 * Returns the command line arguments
 *
 * Returns a list of the command line arguments.
 */
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
// for i in range(0, [args count])
//      res.push([args objectAtIndex:i])
// res
#[cfg(target_os = "ios")]
fn real_args_as_bytes() -> Vec<Vec<u8>> {
    use c_str::CString;
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
        for i in range(0, cnt) {
            let tmp = objc_msgSend(args, objectAtSel, i);
            let utf_c_str: *const libc::c_char =
                mem::transmute(objc_msgSend(tmp, utf8Sel));
            let s = CString::new(utf_c_str, false);
            if s.is_not_null() {
                res.push(Vec::from_slice(s.as_bytes_no_nul()))
            }
        }
    }

    res
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
#[cfg(target_os = "freebsd")]
fn real_args_as_bytes() -> Vec<Vec<u8>> {
    use rt;

    match rt::args::clone() {
        Some(args) => args,
        None => fail!("process arguments not initialized")
    }
}

#[cfg(not(windows))]
fn real_args() -> Vec<String> {
    real_args_as_bytes().move_iter()
                        .map(|v| {
                            str::from_utf8_lossy(v.as_slice()).into_string()
                        }).collect()
}

#[cfg(windows)]
fn real_args() -> Vec<String> {
    use slice;

    let mut nArgs: c_int = 0;
    let lpArgCount: *mut c_int = &mut nArgs;
    let lpCmdLine = unsafe { GetCommandLineW() };
    let szArgList = unsafe { CommandLineToArgvW(lpCmdLine, lpArgCount) };

    let args = Vec::from_fn(nArgs as uint, |i| unsafe {
        // Determine the length of this argument.
        let ptr = *szArgList.offset(i as int);
        let mut len = 0;
        while *ptr.offset(len as int) != 0 { len += 1; }

        // Push it onto the list.
        let opt_s = slice::raw::buf_as_slice(ptr as *const _, len, |buf| {
            str::from_utf16(str::truncate_utf16_at_nul(buf))
        });
        opt_s.expect("CommandLineToArgvW returned invalid UTF-16")
    });

    unsafe {
        LocalFree(szArgList as *mut c_void);
    }

    return args
}

#[cfg(windows)]
fn real_args_as_bytes() -> Vec<Vec<u8>> {
    real_args().move_iter().map(|s| s.into_bytes()).collect()
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
/// The arguments are interpreted as utf-8, with invalid bytes replaced with \uFFFD.
/// See `str::from_utf8_lossy` for details.
pub fn args() -> Vec<String> {
    real_args()
}

/// Returns the arguments which this program was started with (normally passed
/// via the command line) as byte vectors.
pub fn args_as_bytes() -> Vec<Vec<u8>> {
    real_args_as_bytes()
}

#[cfg(target_os = "macos")]
extern {
    // These functions are in crt_externs.h.
    pub fn _NSGetArgc() -> *mut c_int;
    pub fn _NSGetArgv() -> *mut *mut *mut c_char;
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

/// Returns the page size of the current architecture in bytes.
#[cfg(unix)]
pub fn page_size() -> uint {
    unsafe {
        libc::sysconf(libc::_SC_PAGESIZE) as uint
    }
}

/// Returns the page size of the current architecture in bytes.
#[cfg(windows)]
pub fn page_size() -> uint {
    use mem;
    unsafe {
        let mut info = mem::zeroed();
        libc::GetSystemInfo(&mut info);

        return info.dwPageSize as uint;
    }
}

/// A memory mapped file or chunk of memory. This is a very system-specific
/// interface to the OS's memory mapping facilities (`mmap` on POSIX,
/// `VirtualAlloc`/`CreateFileMapping` on win32). It makes no attempt at
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
    /// ## The following are win32-specific
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
        let len = round_up(min_len, page_size());

        for &o in options.iter() {
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
        page_size()
    }
}

#[cfg(unix)]
impl Drop for MemoryMap {
    /// Unmap the mapping. Fails the task if `munmap` fails.
    fn drop(&mut self) {
        if self.len == 0 { /* workaround for dummy_stack */ return; }

        unsafe {
            // `munmap` only fails due to logic errors
            libc::munmap(self.data as *mut c_void, self.len as libc::size_t);
        }
    }
}

#[cfg(windows)]
impl MemoryMap {
    /// Create a new mapping with the given `options`, at least `min_len` bytes long.
    pub fn new(min_len: uint, options: &[MapOption]) -> Result<MemoryMap, MapError> {
        use libc::types::os::arch::extra::{LPVOID, DWORD, SIZE_T, HANDLE};

        let mut lpAddress: LPVOID = ptr::mut_null();
        let mut readable = false;
        let mut writable = false;
        let mut executable = false;
        let mut fd: c_int = -1;
        let mut offset: uint = 0;
        let len = round_up(min_len, page_size());

        for &o in options.iter() {
            match o {
                MapReadable => { readable = true; },
                MapWritable => { writable = true; },
                MapExecutable => { executable = true; }
                MapAddr(addr_) => { lpAddress = addr_ as LPVOID; },
                MapFd(fd_) => { fd = fd_; },
                MapOffset(offset_) => { offset = offset_; },
                MapNonStandardFlags(..) => {}
            }
        }

        let flProtect = match (executable, readable, writable) {
            (false, false, false) if fd == -1 => libc::PAGE_NOACCESS,
            (false, true, false) => libc::PAGE_READONLY,
            (false, true, true) => libc::PAGE_READWRITE,
            (true, false, false) if fd == -1 => libc::PAGE_EXECUTE,
            (true, true, false) => libc::PAGE_EXECUTE_READ,
            (true, true, true) => libc::PAGE_EXECUTE_READWRITE,
            _ => return Err(ErrUnsupProt)
        };

        if fd == -1 {
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
                let hFile = libc::get_osfhandle(fd) as HANDLE;
                let mapping = libc::CreateFileMappingW(hFile,
                                                       ptr::mut_null(),
                                                       flProtect,
                                                       0,
                                                       0,
                                                       ptr::null());
                if mapping == ptr::mut_null() {
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
    /// Unmap the mapping. Fails the task if any of `VirtualFree`,
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
pub mod consts {
    pub use os::arch_consts::ARCH;

    pub static FAMILY: &'static str = "unix";

    /// A string describing the specific operating system in use: in this
    /// case, `linux`.
    pub static SYSNAME: &'static str = "linux";

    /// Specifies the filename prefix used for shared libraries on this
    /// platform: in this case, `lib`.
    pub static DLL_PREFIX: &'static str = "lib";

    /// Specifies the filename suffix used for shared libraries on this
    /// platform: in this case, `.so`.
    pub static DLL_SUFFIX: &'static str = ".so";

    /// Specifies the file extension used for shared libraries on this
    /// platform that goes after the dot: in this case, `so`.
    pub static DLL_EXTENSION: &'static str = "so";

    /// Specifies the filename suffix used for executable binaries on this
    /// platform: in this case, the empty string.
    pub static EXE_SUFFIX: &'static str = "";

    /// Specifies the file extension, if any, used for executable binaries
    /// on this platform: in this case, the empty string.
    pub static EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "macos")]
pub mod consts {
    pub use os::arch_consts::ARCH;

    pub static FAMILY: &'static str = "unix";

    /// A string describing the specific operating system in use: in this
    /// case, `macos`.
    pub static SYSNAME: &'static str = "macos";

    /// Specifies the filename prefix used for shared libraries on this
    /// platform: in this case, `lib`.
    pub static DLL_PREFIX: &'static str = "lib";

    /// Specifies the filename suffix used for shared libraries on this
    /// platform: in this case, `.dylib`.
    pub static DLL_SUFFIX: &'static str = ".dylib";

    /// Specifies the file extension used for shared libraries on this
    /// platform that goes after the dot: in this case, `dylib`.
    pub static DLL_EXTENSION: &'static str = "dylib";

    /// Specifies the filename suffix used for executable binaries on this
    /// platform: in this case, the empty string.
    pub static EXE_SUFFIX: &'static str = "";

    /// Specifies the file extension, if any, used for executable binaries
    /// on this platform: in this case, the empty string.
    pub static EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "ios")]
pub mod consts {
    pub use os::arch_consts::ARCH;

    pub static FAMILY: &'static str = "unix";

    /// A string describing the specific operating system in use: in this
    /// case, `ios`.
    pub static SYSNAME: &'static str = "ios";

    /// Specifies the filename suffix used for executable binaries on this
    /// platform: in this case, the empty string.
    pub static EXE_SUFFIX: &'static str = "";

    /// Specifies the file extension, if any, used for executable binaries
    /// on this platform: in this case, the empty string.
    pub static EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "freebsd")]
pub mod consts {
    pub use os::arch_consts::ARCH;

    pub static FAMILY: &'static str = "unix";

    /// A string describing the specific operating system in use: in this
    /// case, `freebsd`.
    pub static SYSNAME: &'static str = "freebsd";

    /// Specifies the filename prefix used for shared libraries on this
    /// platform: in this case, `lib`.
    pub static DLL_PREFIX: &'static str = "lib";

    /// Specifies the filename suffix used for shared libraries on this
    /// platform: in this case, `.so`.
    pub static DLL_SUFFIX: &'static str = ".so";

    /// Specifies the file extension used for shared libraries on this
    /// platform that goes after the dot: in this case, `so`.
    pub static DLL_EXTENSION: &'static str = "so";

    /// Specifies the filename suffix used for executable binaries on this
    /// platform: in this case, the empty string.
    pub static EXE_SUFFIX: &'static str = "";

    /// Specifies the file extension, if any, used for executable binaries
    /// on this platform: in this case, the empty string.
    pub static EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "android")]
pub mod consts {
    pub use os::arch_consts::ARCH;

    pub static FAMILY: &'static str = "unix";

    /// A string describing the specific operating system in use: in this
    /// case, `android`.
    pub static SYSNAME: &'static str = "android";

    /// Specifies the filename prefix used for shared libraries on this
    /// platform: in this case, `lib`.
    pub static DLL_PREFIX: &'static str = "lib";

    /// Specifies the filename suffix used for shared libraries on this
    /// platform: in this case, `.so`.
    pub static DLL_SUFFIX: &'static str = ".so";

    /// Specifies the file extension used for shared libraries on this
    /// platform that goes after the dot: in this case, `so`.
    pub static DLL_EXTENSION: &'static str = "so";

    /// Specifies the filename suffix used for executable binaries on this
    /// platform: in this case, the empty string.
    pub static EXE_SUFFIX: &'static str = "";

    /// Specifies the file extension, if any, used for executable binaries
    /// on this platform: in this case, the empty string.
    pub static EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "win32")]
pub mod consts {
    pub use os::arch_consts::ARCH;

    pub static FAMILY: &'static str = "windows";

    /// A string describing the specific operating system in use: in this
    /// case, `win32`.
    pub static SYSNAME: &'static str = "win32";

    /// Specifies the filename prefix used for shared libraries on this
    /// platform: in this case, the empty string.
    pub static DLL_PREFIX: &'static str = "";

    /// Specifies the filename suffix used for shared libraries on this
    /// platform: in this case, `.dll`.
    pub static DLL_SUFFIX: &'static str = ".dll";

    /// Specifies the file extension used for shared libraries on this
    /// platform that goes after the dot: in this case, `dll`.
    pub static DLL_EXTENSION: &'static str = "dll";

    /// Specifies the filename suffix used for executable binaries on this
    /// platform: in this case, `.exe`.
    pub static EXE_SUFFIX: &'static str = ".exe";

    /// Specifies the file extension, if any, used for executable binaries
    /// on this platform: in this case, `exe`.
    pub static EXE_EXTENSION: &'static str = "exe";
}

#[cfg(target_arch = "x86")]
mod arch_consts {
    pub static ARCH: &'static str = "x86";
}

#[cfg(target_arch = "x86_64")]
mod arch_consts {
    pub static ARCH: &'static str = "x86_64";
}

#[cfg(target_arch = "arm")]
mod arch_consts {
    pub static ARCH: &'static str = "arm";
}

#[cfg(target_arch = "mips")]
mod arch_consts {
    pub static ARCH: &'static str = "mips";
}

#[cfg(target_arch = "mipsel")]
mod arch_consts {
    pub static ARCH: &'static str = "mipsel";
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use c_str::ToCStr;
    use option;
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
        let mut rng = rand::task_rng();
        let n = format!("TEST{}", rng.gen_ascii_chars().take(10u)
                                     .collect::<String>());
        assert!(getenv(n.as_slice()).is_none());
        n
    }

    #[test]
    fn test_num_cpus() {
        assert!(os::num_cpus() > 0);
    }

    #[test]
    fn test_setenv() {
        let n = make_rand_name();
        setenv(n.as_slice(), "VALUE");
        assert_eq!(getenv(n.as_slice()), option::Some("VALUE".to_string()));
    }

    #[test]
    fn test_unsetenv() {
        let n = make_rand_name();
        setenv(n.as_slice(), "VALUE");
        unsetenv(n.as_slice());
        assert_eq!(getenv(n.as_slice()), option::None);
    }

    #[test]
    #[ignore]
    fn test_setenv_overwrite() {
        let n = make_rand_name();
        setenv(n.as_slice(), "1");
        setenv(n.as_slice(), "2");
        assert_eq!(getenv(n.as_slice()), option::Some("2".to_string()));
        setenv(n.as_slice(), "");
        assert_eq!(getenv(n.as_slice()), option::Some("".to_string()));
    }

    // Windows GetEnvironmentVariable requires some extra work to make sure
    // the buffer the variable is copied into is the right size
    #[test]
    #[ignore]
    fn test_getenv_big() {
        let mut s = "".to_string();
        let mut i = 0i;
        while i < 100 {
            s.push_str("aaaaaaaaaa");
            i += 1;
        }
        let n = make_rand_name();
        setenv(n.as_slice(), s.as_slice());
        debug!("{}", s.clone());
        assert_eq!(getenv(n.as_slice()), option::Some(s));
    }

    #[test]
    fn test_self_exe_name() {
        let path = os::self_exe_name();
        assert!(path.is_some());
        let path = path.unwrap();
        debug!("{:?}", path.clone());

        // Hard to test this function
        assert!(path.is_absolute());
    }

    #[test]
    fn test_self_exe_path() {
        let path = os::self_exe_path();
        assert!(path.is_some());
        let path = path.unwrap();
        debug!("{:?}", path.clone());

        // Hard to test this function
        assert!(path.is_absolute());
    }

    #[test]
    #[ignore]
    fn test_env_getenv() {
        let e = env();
        assert!(e.len() > 0u);
        for p in e.iter() {
            let (n, v) = (*p).clone();
            debug!("{:?}", n.clone());
            let v2 = getenv(n.as_slice());
            // MingW seems to set some funky environment variables like
            // "=C:=C:\MinGW\msys\1.0\bin" and "!::=::\" that are returned
            // from env() but not visible from getenv().
            assert!(v2.is_none() || v2 == option::Some(v));
        }
    }

    #[test]
    fn test_env_set_get_huge() {
        let n = make_rand_name();
        let s = "x".repeat(10000).to_string();
        setenv(n.as_slice(), s.as_slice());
        assert_eq!(getenv(n.as_slice()), Some(s));
        unsetenv(n.as_slice());
        assert_eq!(getenv(n.as_slice()), None);
    }

    #[test]
    fn test_env_setenv() {
        let n = make_rand_name();

        let mut e = env();
        setenv(n.as_slice(), "VALUE");
        assert!(!e.contains(&(n.clone(), "VALUE".to_string())));

        e = env();
        assert!(e.contains(&(n, "VALUE".to_string())));
    }

    #[test]
    fn test() {
        assert!((!Path::new("test-path").is_absolute()));

        let cwd = getcwd();
        debug!("Current working directory: {}", cwd.display());

        debug!("{:?}", make_absolute(&Path::new("test-path")));
        debug!("{:?}", make_absolute(&Path::new("/usr/bin")));
    }

    #[test]
    #[cfg(unix)]
    fn homedir() {
        let oldhome = getenv("HOME");

        setenv("HOME", "/home/MountainView");
        assert!(os::homedir() == Some(Path::new("/home/MountainView")));

        setenv("HOME", "");
        assert!(os::homedir().is_none());

        for s in oldhome.iter() {
            setenv("HOME", s.as_slice());
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

        for s in oldhome.iter() {
            setenv("HOME", s.as_slice());
        }
        for s in olduserprofile.iter() {
            setenv("USERPROFILE", s.as_slice());
        }
    }

    #[test]
    fn memory_map_rw() {
        use result::{Ok, Err};

        let chunk = match os::MemoryMap::new(16, [
            os::MapReadable,
            os::MapWritable
        ]) {
            Ok(chunk) => chunk,
            Err(msg) => fail!("{}", msg)
        };
        assert!(chunk.len >= 16);

        unsafe {
            *chunk.data = 0xBE;
            assert!(*chunk.data == 0xBE);
        }
    }

    #[test]
    fn memory_map_file() {
        use result::{Ok, Err};
        use os::*;
        use libc::*;
        use io::fs;

        #[cfg(unix)]
        fn lseek_(fd: c_int, size: uint) {
            unsafe {
                assert!(lseek(fd, size as off_t, SEEK_SET) == size as off_t);
            }
        }
        #[cfg(windows)]
        fn lseek_(fd: c_int, size: uint) {
           unsafe {
               assert!(lseek(fd, size as c_long, SEEK_SET) == size as c_long);
           }
        }

        let mut path = tmpdir();
        path.push("mmap_file.tmp");
        let size = MemoryMap::granularity() * 2;

        let fd = unsafe {
            let fd = path.with_c_str(|path| {
                open(path, O_CREAT | O_RDWR | O_TRUNC, S_IRUSR | S_IWUSR)
            });
            lseek_(fd, size);
            "x".with_c_str(|x| assert!(write(fd, x as *const c_void, 1) == 1));
            fd
        };
        let chunk = match MemoryMap::new(size / 2, [
            MapReadable,
            MapWritable,
            MapFd(fd),
            MapOffset(size / 2)
        ]) {
            Ok(chunk) => chunk,
            Err(msg) => fail!("{}", msg)
        };
        assert!(chunk.len > 0);

        unsafe {
            *chunk.data = 0xbe;
            assert!(*chunk.data == 0xbe);
            close(fd);
        }
        drop(chunk);

        fs::unlink(&path).unwrap();
    }

    #[test]
    #[cfg(windows)]
    fn split_paths_windows() {
        fn check_parse(unparsed: &str, parsed: &[&str]) -> bool {
            split_paths(unparsed) ==
                parsed.iter().map(|s| Path::new(*s)).collect()
        }

        assert!(check_parse("", [""]));
        assert!(check_parse(r#""""#, [""]));
        assert!(check_parse(";;", ["", "", ""]));
        assert!(check_parse(r"c:\", [r"c:\"]));
        assert!(check_parse(r"c:\;", [r"c:\", ""]));
        assert!(check_parse(r"c:\;c:\Program Files\",
                            [r"c:\", r"c:\Program Files\"]));
        assert!(check_parse(r#"c:\;c:\"foo"\"#, [r"c:\", r"c:\foo\"]));
        assert!(check_parse(r#"c:\;c:\"foo;bar"\;c:\baz"#,
                            [r"c:\", r"c:\foo;bar\", r"c:\baz"]));
    }

    #[test]
    #[cfg(unix)]
    fn split_paths_unix() {
        fn check_parse(unparsed: &str, parsed: &[&str]) -> bool {
            split_paths(unparsed) ==
                parsed.iter().map(|s| Path::new(*s)).collect()
        }

        assert!(check_parse("", [""]));
        assert!(check_parse("::", ["", "", ""]));
        assert!(check_parse("/", ["/"]));
        assert!(check_parse("/:", ["/", ""]));
        assert!(check_parse("/:/usr/local", ["/", "/usr/local"]));
    }

    #[test]
    #[cfg(unix)]
    fn join_paths_unix() {
        fn test_eq(input: &[&str], output: &str) -> bool {
            join_paths(input).unwrap().as_slice() == output.as_bytes()
        }

        assert!(test_eq([], ""));
        assert!(test_eq(["/bin", "/usr/bin", "/usr/local/bin"],
                        "/bin:/usr/bin:/usr/local/bin"));
        assert!(test_eq(["", "/bin", "", "", "/usr/bin", ""],
                        ":/bin:::/usr/bin:"));
        assert!(join_paths(["/te:st"]).is_err());
    }

    #[test]
    #[cfg(windows)]
    fn join_paths_windows() {
        fn test_eq(input: &[&str], output: &str) -> bool {
            join_paths(input).unwrap().as_slice() == output.as_bytes()
        }

        assert!(test_eq([], ""));
        assert!(test_eq([r"c:\windows", r"c:\"],
                        r"c:\windows;c:\"));
        assert!(test_eq(["", r"c:\windows", "", "", r"c:\", ""],
                        r";c:\windows;;;c:\;"));
        assert!(test_eq([r"c:\te;st", r"c:\"],
                        r#""c:\te;st";c:\"#));
        assert!(join_paths([r#"c:\te"st"#]).is_err());
    }

    // More recursive_mkdir tests are in extra::tempfile
}
