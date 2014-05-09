// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
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

#![allow(missing_doc)]

use clone::Clone;
use container::Container;
use libc;
use libc::{c_char, c_void, c_int};
use option::{Some, None, Option};
use os;
use ops::Drop;
use result::{Err, Ok, Result};
use ptr;
use str;
use str::{Str, StrSlice, StrAllocating};
use fmt;
use sync::atomics::{AtomicInt, INIT_ATOMIC_INT, SeqCst};
use path::{Path, GenericPath};
use iter::Iterator;
use slice::{Vector, CloneableVector, ImmutableVector, MutableVector, OwnedVector};
use ptr::RawPtr;
use vec::Vec;

#[cfg(unix)]
use c_str::ToCStr;
#[cfg(windows)]
use str::OwnedStr;

/// Delegates to the libc close() function, returning the same return value.
pub fn close(fd: int) -> int {
    unsafe {
        libc::close(fd as c_int) as int
    }
}

pub static TMPBUF_SZ : uint = 1000u;
static BUF_BYTES : uint = 2048u;

/// Returns the current working directory.
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

/// Returns the current working directory.
#[cfg(windows)]
pub fn getcwd() -> Path {
    use libc::DWORD;
    use libc::GetCurrentDirectoryW;
    use option::Expect;

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
    use iter::Iterator;
    use libc::types::os::arch::extra::DWORD;
    use libc;
    use option::{None, Option, Expect};
    use option;
    use os::TMPBUF_SZ;
    use slice::{MutableVector, ImmutableVector, OwnedVector};
    use str::{StrSlice, StrAllocating};
    use str;
    use vec::Vec;

    pub fn fill_utf16_buf_and_decode(f: |*mut u16, DWORD| -> DWORD)
        -> Option<~str> {

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

    pub fn as_utf16_p<T>(s: &str, f: |*u16| -> T) -> T {
        let mut t = s.to_utf16().move_iter().collect::<Vec<u16>>();
        // Null terminate before passing on.
        t.push(0u16);
        f(t.as_ptr())
    }
}

/*
Accessing environment variables is not generally threadsafe.
Serialize access through a global lock.
*/
fn with_env_lock<T>(f: || -> T) -> T {
    use unstable::mutex::{StaticNativeMutex, NATIVE_MUTEX_INIT};

    static mut lock: StaticNativeMutex = NATIVE_MUTEX_INIT;

    unsafe {
        let _guard = lock.lock();
        f()
    }
}

/// Returns a vector of (variable, value) pairs for all the environment
/// variables of the current process.
///
/// Invalid UTF-8 bytes are replaced with \uFFFD. See `str::from_utf8_lossy()`
/// for details.
pub fn env() -> Vec<(~str,~str)> {
    env_as_bytes().move_iter().map(|(k,v)| {
        let k = str::from_utf8_lossy(k).into_owned();
        let v = str::from_utf8_lossy(v).into_owned();
        (k,v)
    }).collect()
}

/// Returns a vector of (variable, value) byte-vector pairs for all the
/// environment variables of the current process.
pub fn env_as_bytes() -> Vec<(~[u8],~[u8])> {
    unsafe {
        #[cfg(windows)]
        unsafe fn get_env_pairs() -> Vec<~[u8]> {
            use c_str;

            use libc::funcs::extra::kernel32::{
                GetEnvironmentStringsA,
                FreeEnvironmentStringsA
            };
            let ch = GetEnvironmentStringsA();
            if ch as uint == 0 {
                fail!("os::env() failure getting env string from OS: {}",
                       os::last_os_error());
            }
            let mut result = Vec::new();
            c_str::from_c_multistring(ch as *c_char, None, |cstr| {
                result.push(cstr.as_bytes_no_nul().to_owned());
            });
            FreeEnvironmentStringsA(ch);
            result
        }
        #[cfg(unix)]
        unsafe fn get_env_pairs() -> Vec<~[u8]> {
            use c_str::CString;

            extern {
                fn rust_env_pairs() -> **c_char;
            }
            let environ = rust_env_pairs();
            if environ as uint == 0 {
                fail!("os::env() failure getting env string from OS: {}",
                       os::last_os_error());
            }
            let mut result = Vec::new();
            ptr::array_each(environ, |e| {
                let env_pair = CString::new(e, false).as_bytes_no_nul().to_owned();
                result.push(env_pair);
            });
            result
        }

        fn env_convert(input: Vec<~[u8]>) -> Vec<(~[u8], ~[u8])> {
            let mut pairs = Vec::new();
            for p in input.iter() {
                let mut it = p.splitn(1, |b| *b == '=' as u8);
                let key = it.next().unwrap().to_owned();
                let val = it.next().unwrap_or(&[]).to_owned();
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
pub fn getenv(n: &str) -> Option<~str> {
    getenv_as_bytes(n).map(|v| str::from_utf8_lossy(v).into_owned())
}

#[cfg(unix)]
/// Fetches the environment variable `n` byte vector from the current process,
/// returning None if the variable isn't set.
///
/// # Failure
///
/// Fails if `n` has any interior NULs.
pub fn getenv_as_bytes(n: &str) -> Option<~[u8]> {
    use c_str::CString;

    unsafe {
        with_env_lock(|| {
            let s = n.with_c_str(|buf| libc::getenv(buf));
            if s.is_null() {
                None
            } else {
                Some(CString::new(s, false).as_bytes_no_nul().to_owned())
            }
        })
    }
}

#[cfg(windows)]
/// Fetches the environment variable `n` from the current process, returning
/// None if the variable isn't set.
pub fn getenv(n: &str) -> Option<~str> {
    unsafe {
        with_env_lock(|| {
            use os::win32::{as_utf16_p, fill_utf16_buf_and_decode};
            as_utf16_p(n, |u| {
                fill_utf16_buf_and_decode(|buf, sz| {
                    libc::GetEnvironmentVariableW(u, buf, sz)
                })
            })
        })
    }
}

#[cfg(windows)]
/// Fetches the environment variable `n` byte vector from the current process,
/// returning None if the variable isn't set.
pub fn getenv_as_bytes(n: &str) -> Option<~[u8]> {
    getenv(n).map(|s| s.into_bytes())
}


#[cfg(unix)]
/// Sets the environment variable `n` to the value `v` for the currently running
/// process
///
/// # Failure
///
/// Fails if `n` or `v` have any interior NULs.
pub fn setenv(n: &str, v: &str) {
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
/// Sets the environment variable `n` to the value `v` for the currently running
/// process
pub fn setenv(n: &str, v: &str) {
    unsafe {
        with_env_lock(|| {
            use os::win32::as_utf16_p;
            as_utf16_p(n, |nbuf| {
                as_utf16_p(v, |vbuf| {
                    libc::SetEnvironmentVariableW(nbuf, vbuf);
                })
            })
        })
    }
}

/// Remove a variable from the environment entirely
///
/// # Failure
///
/// Fails (on unix) if `n` has any interior NULs.
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
        unsafe {
            with_env_lock(|| {
                use os::win32::as_utf16_p;
                as_utf16_p(n, |nbuf| {
                    libc::SetEnvironmentVariableW(nbuf, ptr::null());
                })
            })
        }
    }

    _unsetenv(n);
}

/// A low-level OS in-memory pipe.
pub struct Pipe {
    /// A file descriptor representing the reading end of the pipe. Data written
    /// on the `out` file descriptor can be read from this file descriptor.
    pub input: c_int,
    /// A file descriptor representing the write end of the pipe. Data written
    /// to this file descriptor can be read from the `input` file descriptor.
    pub out: c_int,
}

/// Creates a new low-level OS in-memory pipe.
#[cfg(unix)]
pub fn pipe() -> Pipe {
    unsafe {
        let mut fds = Pipe {input: 0,
                            out: 0};
        assert_eq!(libc::pipe(&mut fds.input), 0);
        return Pipe {input: fds.input, out: fds.out};
    }
}

/// Creates a new low-level OS in-memory pipe.
#[cfg(windows)]
pub fn pipe() -> Pipe {
    unsafe {
        // Windows pipes work subtly differently than unix pipes, and their
        // inheritance has to be handled in a different way that I do not
        // fully understand. Here we explicitly make the pipe non-inheritable,
        // which means to pass it to a subprocess they need to be duplicated
        // first, as in std::run.
        let mut fds = Pipe {input: 0,
                    out: 0};
        let res = libc::pipe(&mut fds.input, 1024 as ::libc::c_uint,
                             (libc::O_BINARY | libc::O_NOINHERIT) as c_int);
        assert_eq!(res, 0);
        assert!((fds.input != -1 && fds.input != 0 ));
        assert!((fds.out != -1 && fds.input != 0));
        return Pipe {input: fds.input, out: fds.out};
    }
}

/// Returns the proper dll filename for the given basename of a file.
pub fn dll_filename(base: &str) -> ~str {
    format!("{}{}{}", consts::DLL_PREFIX, base, consts::DLL_SUFFIX)
}

/// Optionally returns the filesystem path of the current executable which is
/// running. If any failure occurs, None is returned.
pub fn self_exe_name() -> Option<Path> {

    #[cfg(target_os = "freebsd")]
    fn load_self() -> Option<Vec<u8>> {
        unsafe {
            use libc::funcs::bsd44::*;
            use libc::consts::os::extra::*;
            let mib = box [CTL_KERN as c_int,
                        KERN_PROC as c_int,
                        KERN_PROC_PATHNAME as c_int, -1 as c_int];
            let mut sz: libc::size_t = 0;
            let err = sysctl(mib.as_ptr(), mib.len() as ::libc::c_uint,
                             ptr::mut_null(), &mut sz, ptr::null(),
                             0u as libc::size_t);
            if err != 0 { return None; }
            if sz == 0 { return None; }
            let mut v: Vec<u8> = Vec::with_capacity(sz as uint);
            let err = sysctl(mib.as_ptr(), mib.len() as ::libc::c_uint,
                             v.as_mut_ptr() as *mut c_void, &mut sz, ptr::null(),
                             0u as libc::size_t);
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
            }).map(|s| s.into_strbuf().into_bytes())
        }
    }

    load_self().and_then(Path::new_opt)
}

/// Optionally returns the filesystem path to the current executable which is
/// running. Like self_exe_name() but without the binary's name.
/// If any failure occurs, None is returned.
pub fn self_exe_path() -> Option<Path> {
    self_exe_name().map(|mut p| { p.pop(); p })
}

/**
 * Returns the path to the user's home directory, if known.
 *
 * On Unix, returns the value of the 'HOME' environment variable if it is set
 * and not equal to the empty string.
 *
 * On Windows, returns the value of the 'HOME' environment variable if it is
 * set and not equal to the empty string. Otherwise, returns the value of the
 * 'USERPROFILE' environment variable if it is set and not equal to the empty
 * string.
 *
 * Otherwise, homedir returns option::none.
 */
pub fn homedir() -> Option<Path> {
    // FIXME (#7188): getenv needs a ~[u8] variant
    return match getenv("HOME") {
        Some(ref p) if !p.is_empty() => Path::new_opt(p.as_slice()),
        _ => secondary()
    };

    #[cfg(unix)]
    fn secondary() -> Option<Path> {
        None
    }

    #[cfg(windows)]
    fn secondary() -> Option<Path> {
        getenv("USERPROFILE").and_then(|p| {
            if !p.is_empty() {
                Path::new_opt(p)
            } else {
                None
            }
        })
    }
}

/**
 * Returns the path to a temporary directory.
 *
 * On Unix, returns the value of the 'TMPDIR' environment variable if it is
 * set and non-empty and '/tmp' otherwise.
 * On Android, there is no global temporary folder (it is usually allocated
 * per-app), hence returns '/data/tmp' which is commonly used.
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
        if cfg!(target_os = "android") {
            Path::new("/data/tmp")
        } else {
            getenv_nonempty("TMPDIR").unwrap_or(Path::new("/tmp"))
        }
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
        unsafe {
            use os::win32::as_utf16_p;
            return as_utf16_p(p.as_str().unwrap(), |buf| {
                libc::SetCurrentDirectoryW(buf) != (0 as libc::BOOL)
            });
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
    #[cfg(target_os = "freebsd")]
    fn errno_location() -> *c_int {
        extern {
            fn __error() -> *c_int;
        }
        unsafe {
            __error()
        }
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    fn errno_location() -> *c_int {
        extern {
            fn __errno_location() -> *c_int;
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
pub fn error_string(errnum: uint) -> ~str {
    return strerror(errnum);

    #[cfg(unix)]
    fn strerror(errnum: uint) -> ~str {
        #[cfg(target_os = "macos")]
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

            str::raw::from_c_str(p as *c_char)
        }
    }

    #[cfg(windows)]
    fn strerror(errnum: uint) -> ~str {
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
                              args: *c_void)
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
pub fn last_os_error() -> ~str {
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
unsafe fn load_argc_and_argv(argc: int, argv: **c_char) -> Vec<~[u8]> {
    use c_str::CString;

    Vec::from_fn(argc as uint, |i| {
        CString::new(*argv.offset(i as int), false).as_bytes_no_nul().to_owned()
    })
}

/**
 * Returns the command line arguments
 *
 * Returns a list of the command line arguments.
 */
#[cfg(target_os = "macos")]
fn real_args_as_bytes() -> Vec<~[u8]> {
    unsafe {
        let (argc, argv) = (*_NSGetArgc() as int,
                            *_NSGetArgv() as **c_char);
        load_argc_and_argv(argc, argv)
    }
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
#[cfg(target_os = "freebsd")]
fn real_args_as_bytes() -> Vec<~[u8]> {
    use rt;

    match rt::args::clone() {
        Some(args) => args,
        None => fail!("process arguments not initialized")
    }
}

#[cfg(not(windows))]
fn real_args() -> Vec<~str> {
    real_args_as_bytes().move_iter().map(|v| str::from_utf8_lossy(v).into_owned()).collect()
}

#[cfg(windows)]
fn real_args() -> Vec<~str> {
    use slice;
    use option::Expect;

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
        let opt_s = slice::raw::buf_as_slice(ptr, len, |buf| {
            str::from_utf16(str::truncate_utf16_at_nul(buf))
        });
        opt_s.expect("CommandLineToArgvW returned invalid UTF-16")
    });

    unsafe {
        LocalFree(szArgList as *c_void);
    }

    return args
}

#[cfg(windows)]
fn real_args_as_bytes() -> Vec<~[u8]> {
    real_args().move_iter().map(|s| s.into_bytes()).collect()
}

type LPCWSTR = *u16;

#[cfg(windows)]
#[link_name="kernel32"]
extern "system" {
    fn GetCommandLineW() -> LPCWSTR;
    fn LocalFree(ptr: *c_void);
}

#[cfg(windows)]
#[link_name="shell32"]
extern "system" {
    fn CommandLineToArgvW(lpCmdLine: LPCWSTR, pNumArgs: *mut c_int) -> **u16;
}

/// Returns the arguments which this program was started with (normally passed
/// via the command line).
///
/// The arguments are interpreted as utf-8, with invalid bytes replaced with \uFFFD.
/// See `str::from_utf8_lossy` for details.
pub fn args() -> Vec<~str> {
    real_args()
}

/// Returns the arguments which this program was started with (normally passed
/// via the command line) as byte vectors.
pub fn args_as_bytes() -> Vec<~[u8]> {
    real_args_as_bytes()
}

#[cfg(target_os = "macos")]
extern {
    // These functions are in crt_externs.h.
    pub fn _NSGetArgc() -> *c_int;
    pub fn _NSGetArgv() -> ***c_char;
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
        let mut info = mem::uninit();
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
    /// Pointer to the memory created or modified by this map.
    pub data: *mut u8,
    /// Number of bytes this map applies to
    pub len: uint,
    /// Type of mapping
    pub kind: MemoryMapKind,
}

/// Type of memory map
pub enum MemoryMapKind {
    /// Virtual memory map. Usually used to change the permissions of a given
    /// chunk of memory.  Corresponds to `VirtualAlloc` on Windows.
    MapFile(*u8),
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
    MapAddr(*u8),
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
                return write!(out.buf, "Unknown error = {}", code)
            },
            ErrVirtualAlloc(code) => {
                return write!(out.buf, "VirtualAlloc failure = {}", code)
            },
            ErrCreateFileMappingW(code) => {
                return write!(out.buf, "CreateFileMappingW failure = {}", code)
            },
            ErrMapViewOfFile(code) => {
                return write!(out.buf, "MapViewOfFile failure = {}", code)
            }
        };
        write!(out.buf, "{}", str)
    }
}

#[cfg(unix)]
impl MemoryMap {
    /// Create a new mapping with the given `options`, at least `min_len` bytes
    /// long. `min_len` must be greater than zero; see the note on
    /// `ErrZeroLength`.
    pub fn new(min_len: uint, options: &[MapOption]) -> Result<MemoryMap, MapError> {
        use libc::off_t;
        use cmp::Equiv;

        if min_len == 0 {
            return Err(ErrZeroLength)
        }
        let mut addr: *u8 = ptr::null();
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
            libc::mmap(addr as *c_void, len as libc::size_t, prot, flags, fd,
                       offset)
        };
        if r.equiv(&libc::MAP_FAILED) {
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
            // FIXME: what to do if this fails?
            let _ = libc::munmap(self.data as *c_void, self.len as libc::size_t);
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
                       kind: MapFile(mapping as *u8)
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
            let mut info = mem::uninit();
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

#[cfg(target_os = "linux")]
pub mod consts {
    pub use std::os::arch_consts::ARCH;

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
    pub use std::os::arch_consts::ARCH;

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

#[cfg(target_os = "freebsd")]
pub mod consts {
    pub use std::os::arch_consts::ARCH;

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
    pub use std::os::arch_consts::ARCH;

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
    pub use std::os::arch_consts::ARCH;

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


#[cfg(test)]
mod tests {
    use prelude::*;
    use c_str::ToCStr;
    use option;
    use os::{env, getcwd, getenv, make_absolute, args};
    use os::{setenv, unsetenv};
    use os;
    use rand::Rng;
    use rand;

    #[test]
    pub fn last_os_error() {
        debug!("{}", os::last_os_error());
    }

    #[test]
    pub fn test_args() {
        let a = args();
        assert!(a.len() >= 1);
    }

    fn make_rand_name() -> ~str {
        let mut rng = rand::task_rng();
        let n = "TEST".to_owned() + rng.gen_ascii_str(10u);
        assert!(getenv(n).is_none());
        n
    }

    #[test]
    fn test_setenv() {
        let n = make_rand_name();
        setenv(n, "VALUE");
        assert_eq!(getenv(n), option::Some("VALUE".to_owned()));
    }

    #[test]
    fn test_unsetenv() {
        let n = make_rand_name();
        setenv(n, "VALUE");
        unsetenv(n);
        assert_eq!(getenv(n), option::None);
    }

    #[test]
    #[ignore]
    fn test_setenv_overwrite() {
        let n = make_rand_name();
        setenv(n, "1");
        setenv(n, "2");
        assert_eq!(getenv(n), option::Some("2".to_owned()));
        setenv(n, "");
        assert_eq!(getenv(n), option::Some("".to_owned()));
    }

    // Windows GetEnvironmentVariable requires some extra work to make sure
    // the buffer the variable is copied into is the right size
    #[test]
    #[ignore]
    fn test_getenv_big() {
        let mut s = "".to_owned();
        let mut i = 0;
        while i < 100 {
            s = s + "aaaaaaaaaa";
            i += 1;
        }
        let n = make_rand_name();
        setenv(n, s);
        debug!("{}", s.clone());
        assert_eq!(getenv(n), option::Some(s));
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
            let v2 = getenv(n);
            // MingW seems to set some funky environment variables like
            // "=C:=C:\MinGW\msys\1.0\bin" and "!::=::\" that are returned
            // from env() but not visible from getenv().
            assert!(v2.is_none() || v2 == option::Some(v));
        }
    }

    #[test]
    fn test_env_set_get_huge() {
        let n = make_rand_name();
        let s = "x".repeat(10000);
        setenv(n, s);
        assert_eq!(getenv(n), Some(s));
        unsetenv(n);
        assert_eq!(getenv(n), None);
    }

    #[test]
    fn test_env_setenv() {
        let n = make_rand_name();

        let mut e = env();
        setenv(n, "VALUE");
        assert!(!e.contains(&(n.clone(), "VALUE".to_owned())));

        e = env();
        assert!(e.contains(&(n, "VALUE".to_owned())));
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

        for s in oldhome.iter() { setenv("HOME", *s) }
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

        for s in oldhome.iter() { setenv("HOME", *s) }
        for s in olduserprofile.iter() { setenv("USERPROFILE", *s) }
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
            "x".with_c_str(|x| assert!(write(fd, x as *c_void, 1) == 1));
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

    // More recursive_mkdir tests are in extra::tempfile
}
