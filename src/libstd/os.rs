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

#[allow(missing_doc)];

use c_str::ToCStr;
use clone::Clone;
use container::Container;
use io;
use iterator::range;
use libc;
use libc::{c_char, c_void, c_int, size_t};
use libc::FILE;
use option::{Some, None};
use os;
use prelude::*;
use ptr;
use str;
use to_str;
use unstable::finally::Finally;
use vec;

pub use libc::fclose;
pub use os::consts::*;

/// Delegates to the libc close() function, returning the same return value.
pub fn close(fd: c_int) -> c_int {
    #[fixed_stack_segment]; #[inline(never)];
    unsafe {
        libc::close(fd)
    }
}

pub mod rustrt {
    use libc::{c_char, c_int};
    use libc;

    extern {
        pub fn rust_path_is_dir(path: *libc::c_char) -> c_int;
        pub fn rust_path_exists(path: *libc::c_char) -> c_int;
    }
}

pub static TMPBUF_SZ : uint = 1000u;
static BUF_BYTES : uint = 2048u;

pub fn getcwd() -> Path {
    #[fixed_stack_segment]; #[inline(never)];
    let mut buf = [0 as libc::c_char, ..BUF_BYTES];
    do buf.as_mut_buf |buf, len| {
        unsafe {
            if libc::getcwd(buf, len as size_t).is_null() {
                fail!()
            }

            Path(str::raw::from_c_str(buf as *c_char))
        }
    }
}

// FIXME: move these to str perhaps? #2620

pub fn fill_charp_buf(f: &fn(*mut c_char, size_t) -> bool) -> Option<~str> {
    let mut buf = [0 as c_char, .. TMPBUF_SZ];
    do buf.as_mut_buf |b, sz| {
        if f(b, sz as size_t) {
            unsafe {
                Some(str::raw::from_c_str(b as *c_char))
            }
        } else {
            None
        }
    }
}

#[cfg(windows)]
pub mod win32 {
    use libc;
    use vec;
    use str;
    use option::{None, Option};
    use option;
    use os::TMPBUF_SZ;
    use libc::types::os::arch::extra::DWORD;

    pub fn fill_utf16_buf_and_decode(f: &fn(*mut u16, DWORD) -> DWORD)
        -> Option<~str> {
        #[fixed_stack_segment]; #[inline(never)];

        unsafe {
            let mut n = TMPBUF_SZ as DWORD;
            let mut res = None;
            let mut done = false;
            while !done {
                let mut k: DWORD = 0;
                let mut buf = vec::from_elem(n as uint, 0u16);
                do buf.as_mut_buf |b, _sz| {
                    k = f(b, TMPBUF_SZ as DWORD);
                    if k == (0 as DWORD) {
                        done = true;
                    } else if (k == n &&
                               libc::GetLastError() ==
                               libc::ERROR_INSUFFICIENT_BUFFER as DWORD) {
                        n *= (2 as DWORD);
                    } else {
                        done = true;
                    }
                }
                if k != 0 && done {
                    let sub = buf.slice(0, k as uint);
                    res = option::Some(str::from_utf16(sub));
                }
            }
            return res;
        }
    }

    pub fn as_utf16_p<T>(s: &str, f: &fn(*u16) -> T) -> T {
        let mut t = s.to_utf16();
        // Null terminate before passing on.
        t.push(0u16);
        t.as_imm_buf(|buf, _len| f(buf))
    }
}

#[cfg(stage0)]
mod macro_hack {
#[macro_escape];
macro_rules! externfn(
    (fn $name:ident ()) => (
        extern {
            fn $name();
        }
    )
)
}

/*
Accessing environment variables is not generally threadsafe.
Serialize access through a global lock.
*/
fn with_env_lock<T>(f: &fn() -> T) -> T {
    use unstable::finally::Finally;

    unsafe {
        return do (|| {
            rust_take_env_lock();
            f()
        }).finally {
            rust_drop_env_lock();
        };
    }

    externfn!(fn rust_take_env_lock());
    externfn!(fn rust_drop_env_lock());
}

/// Returns a vector of (variable, value) pairs for all the environment
/// variables of the current process.
pub fn env() -> ~[(~str,~str)] {
    unsafe {
        #[cfg(windows)]
        unsafe fn get_env_pairs() -> ~[~str] {
            #[fixed_stack_segment]; #[inline(never)];

            use libc::funcs::extra::kernel32::{
                GetEnvironmentStringsA,
                FreeEnvironmentStringsA
            };
            let ch = GetEnvironmentStringsA();
            if (ch as uint == 0) {
                fail!("os::env() failure getting env string from OS: %s", os::last_os_error());
            }
            let mut curr_ptr: uint = ch as uint;
            let mut result = ~[];
            while(*(curr_ptr as *libc::c_char) != 0 as libc::c_char) {
                let env_pair = str::raw::from_c_str(
                    curr_ptr as *libc::c_char);
                result.push(env_pair);
                curr_ptr +=
                    libc::strlen(curr_ptr as *libc::c_char) as uint
                    + 1;
            }
            FreeEnvironmentStringsA(ch);
            result
        }
        #[cfg(unix)]
        unsafe fn get_env_pairs() -> ~[~str] {
            #[fixed_stack_segment]; #[inline(never)];

            extern {
                fn rust_env_pairs() -> **libc::c_char;
            }
            let environ = rust_env_pairs();
            if (environ as uint == 0) {
                fail!("os::env() failure getting env string from OS: %s", os::last_os_error());
            }
            let mut result = ~[];
            ptr::array_each(environ, |e| {
                let env_pair = str::raw::from_c_str(e);
                debug!("get_env_pairs: %s",
                       env_pair);
                result.push(env_pair);
            });
            result
        }

        fn env_convert(input: ~[~str]) -> ~[(~str, ~str)] {
            let mut pairs = ~[];
            for p in input.iter() {
                let vs: ~[&str] = p.splitn_iter('=', 1).collect();
                debug!("splitting: len: %u",
                    vs.len());
                assert_eq!(vs.len(), 2);
                pairs.push((vs[0].to_owned(), vs[1].to_owned()));
            }
            pairs
        }
        do with_env_lock {
            let unparsed_environ = get_env_pairs();
            env_convert(unparsed_environ)
        }
    }
}

#[cfg(unix)]
/// Fetches the environment variable `n` from the current process, returning
/// None if the variable isn't set.
pub fn getenv(n: &str) -> Option<~str> {
    #[fixed_stack_segment]; #[inline(never)];
    unsafe {
        do with_env_lock {
            let s = do n.with_c_str |buf| {
                libc::getenv(buf)
            };
            if s.is_null() {
                None
            } else {
                Some(str::raw::from_c_str(s))
            }
        }
    }
}

#[cfg(windows)]
/// Fetches the environment variable `n` from the current process, returning
/// None if the variable isn't set.
pub fn getenv(n: &str) -> Option<~str> {
    #[fixed_stack_segment]; #[inline(never)];

    unsafe {
        do with_env_lock {
            use os::win32::{as_utf16_p, fill_utf16_buf_and_decode};
            do as_utf16_p(n) |u| {
                do fill_utf16_buf_and_decode() |buf, sz| {
                    libc::GetEnvironmentVariableW(u, buf, sz)
                }
            }
        }
    }
}


#[cfg(unix)]
/// Sets the environment variable `n` to the value `v` for the currently running
/// process
pub fn setenv(n: &str, v: &str) {
    #[fixed_stack_segment]; #[inline(never)];
    unsafe {
        do with_env_lock {
            do n.with_c_str |nbuf| {
                do v.with_c_str |vbuf| {
                    libc::funcs::posix01::unistd::setenv(nbuf, vbuf, 1);
                }
            }
        }
    }
}


#[cfg(windows)]
/// Sets the environment variable `n` to the value `v` for the currently running
/// process
pub fn setenv(n: &str, v: &str) {
    #[fixed_stack_segment]; #[inline(never)];

    unsafe {
        do with_env_lock {
            use os::win32::as_utf16_p;
            do as_utf16_p(n) |nbuf| {
                do as_utf16_p(v) |vbuf| {
                    libc::SetEnvironmentVariableW(nbuf, vbuf);
                }
            }
        }
    }
}

/// Remove a variable from the environment entirely
pub fn unsetenv(n: &str) {
    #[cfg(unix)]
    fn _unsetenv(n: &str) {
        #[fixed_stack_segment]; #[inline(never)];
        unsafe {
            do with_env_lock {
                do n.with_c_str |nbuf| {
                    libc::funcs::posix01::unistd::unsetenv(nbuf);
                }
            }
        }
    }
    #[cfg(windows)]
    fn _unsetenv(n: &str) {
        #[fixed_stack_segment]; #[inline(never)];
        unsafe {
            do with_env_lock {
                use os::win32::as_utf16_p;
                do as_utf16_p(n) |nbuf| {
                    libc::SetEnvironmentVariableW(nbuf, ptr::null());
                }
            }
        }
    }

    _unsetenv(n);
}

pub fn fdopen(fd: c_int) -> *FILE {
    #[fixed_stack_segment]; #[inline(never)];
    do "r".with_c_str |modebuf| {
        unsafe {
            libc::fdopen(fd, modebuf)
        }
    }
}


// fsync related

#[cfg(windows)]
pub fn fsync_fd(fd: c_int, _level: io::fsync::Level) -> c_int {
    #[fixed_stack_segment]; #[inline(never)];
    unsafe {
        use libc::funcs::extra::msvcrt::*;
        return commit(fd);
    }
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
pub fn fsync_fd(fd: c_int, level: io::fsync::Level) -> c_int {
    #[fixed_stack_segment]; #[inline(never)];
    unsafe {
        use libc::funcs::posix01::unistd::*;
        match level {
          io::fsync::FSync
          | io::fsync::FullFSync => return fsync(fd),
          io::fsync::FDataSync => return fdatasync(fd)
        }
    }
}

#[cfg(target_os = "macos")]
pub fn fsync_fd(fd: c_int, level: io::fsync::Level) -> c_int {
    #[fixed_stack_segment]; #[inline(never)];

    unsafe {
        use libc::consts::os::extra::*;
        use libc::funcs::posix88::fcntl::*;
        use libc::funcs::posix01::unistd::*;
        match level {
          io::fsync::FSync => return fsync(fd),
          _ => {
            // According to man fnctl, the ok retval is only specified to be
            // !=-1
            if (fcntl(F_FULLFSYNC as c_int, fd) == -1 as c_int)
                { return -1 as c_int; }
            else
                { return 0 as c_int; }
          }
        }
    }
}

#[cfg(target_os = "freebsd")]
pub fn fsync_fd(fd: c_int, _l: io::fsync::Level) -> c_int {
    #[fixed_stack_segment]; #[inline(never)];

    unsafe {
        use libc::funcs::posix01::unistd::*;
        return fsync(fd);
    }
}

pub struct Pipe {
    input: c_int,
    out: c_int
}

#[cfg(unix)]
pub fn pipe() -> Pipe {
    #[fixed_stack_segment]; #[inline(never)];
    unsafe {
        let mut fds = Pipe {input: 0 as c_int,
                            out: 0 as c_int };
        assert_eq!(libc::pipe(&mut fds.input), (0 as c_int));
        return Pipe {input: fds.input, out: fds.out};
    }
}



#[cfg(windows)]
pub fn pipe() -> Pipe {
    #[fixed_stack_segment]; #[inline(never)];
    unsafe {
        // Windows pipes work subtly differently than unix pipes, and their
        // inheritance has to be handled in a different way that I do not
        // fully understand. Here we explicitly make the pipe non-inheritable,
        // which means to pass it to a subprocess they need to be duplicated
        // first, as in std::run.
        let mut fds = Pipe {input: 0 as c_int,
                    out: 0 as c_int };
        let res = libc::pipe(&mut fds.input, 1024 as ::libc::c_uint,
                             (libc::O_BINARY | libc::O_NOINHERIT) as c_int);
        assert_eq!(res, 0 as c_int);
        assert!((fds.input != -1 as c_int && fds.input != 0 as c_int));
        assert!((fds.out != -1 as c_int && fds.input != 0 as c_int));
        return Pipe {input: fds.input, out: fds.out};
    }
}

fn dup2(src: c_int, dst: c_int) -> c_int {
    #[fixed_stack_segment]; #[inline(never)];
    unsafe {
        libc::dup2(src, dst)
    }
}

/// Returns the proper dll filename for the given basename of a file.
pub fn dll_filename(base: &str) -> ~str {
    fmt!("%s%s%s", DLL_PREFIX, base, DLL_SUFFIX)
}

/// Optionally returns the filesystem path to the current executable which is
/// running. If any failure occurs, None is returned.
pub fn self_exe_path() -> Option<Path> {

    #[cfg(target_os = "freebsd")]
    fn load_self() -> Option<~str> {
        #[fixed_stack_segment]; #[inline(never)];
        unsafe {
            use libc::funcs::bsd44::*;
            use libc::consts::os::extra::*;
            do fill_charp_buf() |buf, sz| {
                let mib = ~[CTL_KERN as c_int,
                           KERN_PROC as c_int,
                           KERN_PROC_PATHNAME as c_int, -1 as c_int];
                let mut sz = sz;
                sysctl(vec::raw::to_ptr(mib), mib.len() as ::libc::c_uint,
                       buf as *mut c_void, &mut sz, ptr::null(),
                       0u as size_t) == (0 as c_int)
            }
        }
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    fn load_self() -> Option<~str> {
        #[fixed_stack_segment]; #[inline(never)];
        unsafe {
            use libc::funcs::posix01::unistd::readlink;

            let mut path = [0 as c_char, .. TMPBUF_SZ];

            do path.as_mut_buf |buf, len| {
                let len = do "/proc/self/exe".with_c_str |proc_self_buf| {
                    readlink(proc_self_buf, buf, len as size_t) as uint
                };

                if len == -1 {
                    None
                } else {
                    Some(str::raw::from_buf_len(buf as *u8, len))
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn load_self() -> Option<~str> {
        #[fixed_stack_segment]; #[inline(never)];
        unsafe {
            do fill_charp_buf() |buf, sz| {
                let mut sz = sz as u32;
                libc::funcs::extra::_NSGetExecutablePath(
                    buf, &mut sz) == (0 as c_int)
            }
        }
    }

    #[cfg(windows)]
    fn load_self() -> Option<~str> {
        #[fixed_stack_segment]; #[inline(never)];
        unsafe {
            use os::win32::fill_utf16_buf_and_decode;
            do fill_utf16_buf_and_decode() |buf, sz| {
                libc::GetModuleFileNameW(0u as libc::DWORD, buf, sz)
            }
        }
    }

    load_self().map_move(|path| Path(path).dir_path())
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
    return match getenv("HOME") {
        Some(ref p) => if !p.is_empty() {
          Some(Path(*p))
        } else {
          secondary()
        },
        None => secondary()
    };

    #[cfg(unix)]
    fn secondary() -> Option<Path> {
        None
    }

    #[cfg(windows)]
    fn secondary() -> Option<Path> {
        do getenv("USERPROFILE").chain |p| {
            if !p.is_empty() {
                Some(Path(p))
            } else {
                None
            }
        }
    }
}

/**
 * Returns the path to a temporary directory.
 *
 * On Unix, returns the value of the 'TMPDIR' environment variable if it is
 * set and non-empty and '/tmp' otherwise.
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
                    Some(Path(x))
                },
            _ => None
        }
    }

    #[cfg(unix)]
    fn lookup() -> Path {
        getenv_nonempty("TMPDIR").unwrap_or_default(Path("/tmp"))
    }

    #[cfg(windows)]
    fn lookup() -> Path {
        getenv_nonempty("TMP").or(
            getenv_nonempty("TEMP").or(
                getenv_nonempty("USERPROFILE").or(
                   getenv_nonempty("WINDIR")))).unwrap_or_default(Path("C:\\Windows"))
    }
}

/// Recursively walk a directory structure
pub fn walk_dir(p: &Path, f: &fn(&Path) -> bool) -> bool {
    let r = list_dir(p);
    r.iter().advance(|q| {
        let path = &p.push(*q);
        f(path) && (!path_is_dir(path) || walk_dir(path, |p| f(p)))
    })
}

/// Indicates whether a path represents a directory
pub fn path_is_dir(p: &Path) -> bool {
    #[fixed_stack_segment]; #[inline(never)];
    unsafe {
        do p.with_c_str |buf| {
            rustrt::rust_path_is_dir(buf) != 0 as c_int
        }
    }
}

/// Indicates whether a path exists
pub fn path_exists(p: &Path) -> bool {
    #[fixed_stack_segment]; #[inline(never)];
    unsafe {
        do p.with_c_str |buf| {
            rustrt::rust_path_exists(buf) != 0 as c_int
        }
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
    if p.is_absolute {
        (*p).clone()
    } else {
        getcwd().push_many(p.components)
    }
}


/// Creates a directory at the specified path
pub fn make_dir(p: &Path, mode: c_int) -> bool {
    return mkdir(p, mode);

    #[cfg(windows)]
    fn mkdir(p: &Path, _mode: c_int) -> bool {
        #[fixed_stack_segment]; #[inline(never)];
        unsafe {
            use os::win32::as_utf16_p;
            // FIXME: turn mode into something useful? #2623
            do as_utf16_p(p.to_str()) |buf| {
                libc::CreateDirectoryW(buf, ptr::mut_null())
                    != (0 as libc::BOOL)
            }
        }
    }

    #[cfg(unix)]
    fn mkdir(p: &Path, mode: c_int) -> bool {
        #[fixed_stack_segment]; #[inline(never)];
        do p.with_c_str |buf| {
            unsafe {
                libc::mkdir(buf, mode as libc::mode_t) == (0 as c_int)
            }
        }
    }
}

/// Creates a directory with a given mode.
/// Returns true iff creation
/// succeeded. Also creates all intermediate subdirectories
/// if they don't already exist, giving all of them the same mode.

// tjc: if directory exists but with different permissions,
// should we return false?
pub fn mkdir_recursive(p: &Path, mode: c_int) -> bool {
    if path_is_dir(p) {
        return true;
    }
    else if p.components.is_empty() {
        return false;
    }
    else if p.components.len() == 1 {
        // No parent directories to create
        path_is_dir(p) || make_dir(p, mode)
    }
    else {
        mkdir_recursive(&p.pop(), mode) && make_dir(p, mode)
    }
}

/// Lists the contents of a directory
pub fn list_dir(p: &Path) -> ~[~str] {
    if p.components.is_empty() && !p.is_absolute() {
        // Not sure what the right behavior is here, but this
        // prevents a bounds check failure later
        return ~[];
    }
    unsafe {
        #[cfg(target_os = "linux")]
        #[cfg(target_os = "android")]
        #[cfg(target_os = "freebsd")]
        #[cfg(target_os = "macos")]
        unsafe fn get_list(p: &Path) -> ~[~str] {
            #[fixed_stack_segment]; #[inline(never)];
            use libc::{dirent_t};
            use libc::{opendir, readdir, closedir};
            extern {
                fn rust_list_dir_val(ptr: *dirent_t) -> *libc::c_char;
            }
            let mut strings = ~[];
            debug!("os::list_dir -- BEFORE OPENDIR");

            let dir_ptr = do p.with_c_str |buf| {
                opendir(buf)
            };

            if (dir_ptr as uint != 0) {
                debug!("os::list_dir -- opendir() SUCCESS");
                let mut entry_ptr = readdir(dir_ptr);
                while (entry_ptr as uint != 0) {
                    strings.push(str::raw::from_c_str(rust_list_dir_val(
                        entry_ptr)));
                    entry_ptr = readdir(dir_ptr);
                }
                closedir(dir_ptr);
            }
            else {
                debug!("os::list_dir -- opendir() FAILURE");
            }
            debug!(
                "os::list_dir -- AFTER -- #: %?",
                     strings.len());
            strings
        }
        #[cfg(windows)]
        unsafe fn get_list(p: &Path) -> ~[~str] {
            #[fixed_stack_segment]; #[inline(never)];
            use libc::consts::os::extra::INVALID_HANDLE_VALUE;
            use libc::{wcslen, free};
            use libc::funcs::extra::kernel32::{
                FindFirstFileW,
                FindNextFileW,
                FindClose,
            };
            use libc::types::os::arch::extra::HANDLE;
            use os::win32::{
                as_utf16_p
            };
            use rt::global_heap::malloc_raw;

            #[nolink]
            extern {
                fn rust_list_dir_wfd_size() -> libc::size_t;
                fn rust_list_dir_wfd_fp_buf(wfd: *libc::c_void) -> *u16;
            }
            fn star(p: &Path) -> Path { p.push("*") }
            do as_utf16_p(star(p).to_str()) |path_ptr| {
                let mut strings = ~[];
                let wfd_ptr = malloc_raw(rust_list_dir_wfd_size() as uint);
                let find_handle = FindFirstFileW(path_ptr, wfd_ptr as HANDLE);
                if find_handle as libc::c_int != INVALID_HANDLE_VALUE {
                    let mut more_files = 1 as libc::c_int;
                    while more_files != 0 {
                        let fp_buf = rust_list_dir_wfd_fp_buf(wfd_ptr);
                        if fp_buf as uint == 0 {
                            fail!("os::list_dir() failure: got null ptr from wfd");
                        }
                        else {
                            let fp_vec = vec::from_buf(
                                fp_buf, wcslen(fp_buf) as uint);
                            let fp_str = str::from_utf16(fp_vec);
                            strings.push(fp_str);
                        }
                        more_files = FindNextFileW(find_handle, wfd_ptr as HANDLE);
                    }
                    FindClose(find_handle);
                    free(wfd_ptr)
                }
                strings
            }
        }
        do get_list(p).move_iter().filter |filename| {
            "." != *filename && ".." != *filename
        }.collect()
    }
}

/**
 * Lists the contents of a directory
 *
 * This version prepends each entry with the directory.
 */
pub fn list_dir_path(p: &Path) -> ~[Path] {
    list_dir(p).map(|f| p.push(*f))
}

/// Removes a directory at the specified path, after removing
/// all its contents. Use carefully!
pub fn remove_dir_recursive(p: &Path) -> bool {
    let mut error_happened = false;
    do walk_dir(p) |inner| {
        if !error_happened {
            if path_is_dir(inner) {
                if !remove_dir_recursive(inner) {
                    error_happened = true;
                }
            }
            else {
                if !remove_file(inner) {
                    error_happened = true;
                }
            }
        }
        true
    };
    // Directory should now be empty
    !error_happened && remove_dir(p)
}

/// Removes a directory at the specified path
pub fn remove_dir(p: &Path) -> bool {
   return rmdir(p);

    #[cfg(windows)]
    fn rmdir(p: &Path) -> bool {
        #[fixed_stack_segment]; #[inline(never)];
        unsafe {
            use os::win32::as_utf16_p;
            return do as_utf16_p(p.to_str()) |buf| {
                libc::RemoveDirectoryW(buf) != (0 as libc::BOOL)
            };
        }
    }

    #[cfg(unix)]
    fn rmdir(p: &Path) -> bool {
        #[fixed_stack_segment]; #[inline(never)];
        do p.with_c_str |buf| {
            unsafe {
                libc::rmdir(buf) == (0 as c_int)
            }
        }
    }
}

/// Changes the current working directory to the specified path, returning
/// whether the change was completed successfully or not.
pub fn change_dir(p: &Path) -> bool {
    return chdir(p);

    #[cfg(windows)]
    fn chdir(p: &Path) -> bool {
        #[fixed_stack_segment]; #[inline(never)];
        unsafe {
            use os::win32::as_utf16_p;
            return do as_utf16_p(p.to_str()) |buf| {
                libc::SetCurrentDirectoryW(buf) != (0 as libc::BOOL)
            };
        }
    }

    #[cfg(unix)]
    fn chdir(p: &Path) -> bool {
        #[fixed_stack_segment]; #[inline(never)];
        do p.with_c_str |buf| {
            unsafe {
                libc::chdir(buf) == (0 as c_int)
            }
        }
    }
}

/// Copies a file from one location to another
pub fn copy_file(from: &Path, to: &Path) -> bool {
    return do_copy_file(from, to);

    #[cfg(windows)]
    fn do_copy_file(from: &Path, to: &Path) -> bool {
        #[fixed_stack_segment]; #[inline(never)];
        unsafe {
            use os::win32::as_utf16_p;
            return do as_utf16_p(from.to_str()) |fromp| {
                do as_utf16_p(to.to_str()) |top| {
                    libc::CopyFileW(fromp, top, (0 as libc::BOOL)) !=
                        (0 as libc::BOOL)
                }
            }
        }
    }

    #[cfg(unix)]
    fn do_copy_file(from: &Path, to: &Path) -> bool {
        #[fixed_stack_segment]; #[inline(never)];
        unsafe {
            let istream = do from.with_c_str |fromp| {
                do "rb".with_c_str |modebuf| {
                    libc::fopen(fromp, modebuf)
                }
            };
            if istream as uint == 0u {
                return false;
            }
            // Preserve permissions
            let from_mode = from.get_mode().expect("copy_file: couldn't get permissions \
                                                    for source file");

            let ostream = do to.with_c_str |top| {
                do "w+b".with_c_str |modebuf| {
                    libc::fopen(top, modebuf)
                }
            };
            if ostream as uint == 0u {
                fclose(istream);
                return false;
            }
            let bufsize = 8192u;
            let mut buf = vec::with_capacity::<u8>(bufsize);
            let mut done = false;
            let mut ok = true;
            while !done {
                do buf.as_mut_buf |b, _sz| {
                  let nread = libc::fread(b as *mut c_void, 1u as size_t,
                                          bufsize as size_t,
                                          istream);
                  if nread > 0 as size_t {
                      if libc::fwrite(b as *c_void, 1u as size_t, nread,
                                      ostream) != nread {
                          ok = false;
                          done = true;
                      }
                  } else {
                      done = true;
                  }
              }
            }
            fclose(istream);
            fclose(ostream);

            // Give the new file the old file's permissions
            if do to.with_c_str |to_buf| {
                libc::chmod(to_buf, from_mode as libc::mode_t)
            } != 0 {
                return false; // should be a condition...
            }
            return ok;
        }
    }
}

/// Deletes an existing file
pub fn remove_file(p: &Path) -> bool {
    return unlink(p);

    #[cfg(windows)]
    fn unlink(p: &Path) -> bool {
        #[fixed_stack_segment]; #[inline(never)];
        unsafe {
            use os::win32::as_utf16_p;
            return do as_utf16_p(p.to_str()) |buf| {
                libc::DeleteFileW(buf) != (0 as libc::BOOL)
            };
        }
    }

    #[cfg(unix)]
    fn unlink(p: &Path) -> bool {
        #[fixed_stack_segment]; #[inline(never)];
        unsafe {
            do p.with_c_str |buf| {
                libc::unlink(buf) == (0 as c_int)
            }
        }
    }
}

/// Renames an existing file or directory
pub fn rename_file(old: &Path, new: &Path) -> bool {
    #[fixed_stack_segment]; #[inline(never)];
    unsafe {
       do old.with_c_str |old_buf| {
            do new.with_c_str |new_buf| {
                libc::rename(old_buf, new_buf) == (0 as c_int)
            }
       }
    }
}

#[cfg(unix)]
/// Returns the platform-specific value of errno
pub fn errno() -> int {
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn errno_location() -> *c_int {
        #[fixed_stack_segment]; #[inline(never)];
        #[nolink]
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
        #[fixed_stack_segment]; #[inline(never)];
        #[nolink]
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
    #[fixed_stack_segment]; #[inline(never)];
    use libc::types::os::arch::extra::DWORD;

    #[cfg(target_arch = "x86")]
    #[link_name = "kernel32"]
    #[abi = "stdcall"]
    extern "stdcall" {
        fn GetLastError() -> DWORD;
    }

    #[cfg(target_arch = "x86_64")]
    #[link_name = "kernel32"]
    extern {
        fn GetLastError() -> DWORD;
    }

    unsafe {
        GetLastError() as uint
    }
}

/// Get a string representing the platform-dependent last error
pub fn last_os_error() -> ~str {
    #[cfg(unix)]
    fn strerror() -> ~str {
        #[cfg(target_os = "macos")]
        #[cfg(target_os = "android")]
        #[cfg(target_os = "freebsd")]
        fn strerror_r(errnum: c_int, buf: *mut c_char, buflen: size_t)
                      -> c_int {
            #[fixed_stack_segment]; #[inline(never)];

            #[nolink]
            extern {
                fn strerror_r(errnum: c_int, buf: *mut c_char, buflen: size_t)
                              -> c_int;
            }
            unsafe {
                strerror_r(errnum, buf, buflen)
            }
        }

        // GNU libc provides a non-compliant version of strerror_r by default
        // and requires macros to instead use the POSIX compliant variant.
        // So we just use __xpg_strerror_r which is always POSIX compliant
        #[cfg(target_os = "linux")]
        fn strerror_r(errnum: c_int, buf: *mut c_char, buflen: size_t) -> c_int {
            #[fixed_stack_segment]; #[inline(never)];
            #[nolink]
            extern {
                fn __xpg_strerror_r(errnum: c_int,
                                    buf: *mut c_char,
                                    buflen: size_t)
                                    -> c_int;
            }
            unsafe {
                __xpg_strerror_r(errnum, buf, buflen)
            }
        }

        let mut buf = [0 as c_char, ..TMPBUF_SZ];

        do buf.as_mut_buf |buf, len| {
            unsafe {
                if strerror_r(errno() as c_int, buf, len as size_t) < 0 {
                    fail!("strerror_r failure");
                }

                str::raw::from_c_str(buf as *c_char)
            }
        }
    }

    #[cfg(windows)]
    fn strerror() -> ~str {
        #[fixed_stack_segment]; #[inline(never)];

        use libc::types::os::arch::extra::DWORD;
        use libc::types::os::arch::extra::LPSTR;
        use libc::types::os::arch::extra::LPVOID;

        #[cfg(target_arch = "x86")]
        #[link_name = "kernel32"]
        #[abi = "stdcall"]
        extern "stdcall" {
            fn FormatMessageA(flags: DWORD,
                              lpSrc: LPVOID,
                              msgId: DWORD,
                              langId: DWORD,
                              buf: LPSTR,
                              nsize: DWORD,
                              args: *c_void)
                              -> DWORD;
        }

        #[cfg(target_arch = "x86_64")]
        #[link_name = "kernel32"]
        extern {
            fn FormatMessageA(flags: DWORD,
                              lpSrc: LPVOID,
                              msgId: DWORD,
                              langId: DWORD,
                              buf: LPSTR,
                              nsize: DWORD,
                              args: *c_void)
                              -> DWORD;
        }

        static FORMAT_MESSAGE_FROM_SYSTEM: DWORD = 0x00001000;
        static FORMAT_MESSAGE_IGNORE_INSERTS: DWORD = 0x00000200;

        // This value is calculated from the macro
        // MAKELANGID(LANG_SYSTEM_DEFAULT, SUBLANG_SYS_DEFAULT)
        let langId = 0x0800 as DWORD;
        let err = errno() as DWORD;

        let mut buf = [0 as c_char, ..TMPBUF_SZ];

        unsafe {
            do buf.as_mut_buf |buf, len| {
                let res = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM |
                                         FORMAT_MESSAGE_IGNORE_INSERTS,
                                         ptr::mut_null(),
                                         err,
                                         langId,
                                         buf,
                                         len as DWORD,
                                         ptr::null());
                if res == 0 {
                    fail!("[%?] FormatMessage failure", errno());
                }
            }

            do buf.as_imm_buf |buf, _len| {
                str::raw::from_c_str(buf)
            }
        }
    }

    strerror()
}

/**
 * Sets the process exit code
 *
 * Sets the exit code returned by the process if all supervised tasks
 * terminate successfully (without failing). If the current root task fails
 * and is supervised by the scheduler then any user-specified exit status is
 * ignored and the process exits with the default failure status
 */
pub fn set_exit_status(code: int) {
    use rt;
    rt::util::set_exit_status(code);
}

unsafe fn load_argc_and_argv(argc: c_int, argv: **c_char) -> ~[~str] {
    let mut args = ~[];
    for i in range(0u, argc as uint) {
        args.push(str::raw::from_c_str(*argv.offset(i as int)));
    }
    args
}

/**
 * Returns the command line arguments
 *
 * Returns a list of the command line arguments.
 */
#[cfg(target_os = "macos")]
fn real_args() -> ~[~str] {
    #[fixed_stack_segment]; #[inline(never)];

    unsafe {
        let (argc, argv) = (*_NSGetArgc() as c_int,
                            *_NSGetArgv() as **c_char);
        load_argc_and_argv(argc, argv)
    }
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
#[cfg(target_os = "freebsd")]
fn real_args() -> ~[~str] {
    use rt;

    match rt::args::clone() {
        Some(args) => args,
        None => fail!("process arguments not initialized")
    }
}

#[cfg(windows)]
fn real_args() -> ~[~str] {
    #[fixed_stack_segment]; #[inline(never)];

    let mut nArgs: c_int = 0;
    let lpArgCount: *mut c_int = &mut nArgs;
    let lpCmdLine = unsafe { GetCommandLineW() };
    let szArgList = unsafe { CommandLineToArgvW(lpCmdLine, lpArgCount) };

    let mut args = ~[];
    for i in range(0u, nArgs as uint) {
        unsafe {
            // Determine the length of this argument.
            let ptr = *szArgList.offset(i as int);
            let mut len = 0;
            while *ptr.offset(len as int) != 0 { len += 1; }

            // Push it onto the list.
            args.push(vec::raw::buf_as_slice(ptr, len,
                                             str::from_utf16));
        }
    }

    unsafe {
        LocalFree(szArgList as *c_void);
    }

    return args;
}

type LPCWSTR = *u16;

#[cfg(windows, target_arch = "x86")]
#[link_name="kernel32"]
#[abi="stdcall"]
extern "stdcall" {
    fn GetCommandLineW() -> LPCWSTR;
    fn LocalFree(ptr: *c_void);
}

#[cfg(windows, target_arch = "x86_64")]
#[link_name="kernel32"]
extern {
    fn GetCommandLineW() -> LPCWSTR;
    fn LocalFree(ptr: *c_void);
}

#[cfg(windows, target_arch = "x86")]
#[link_name="shell32"]
#[abi="stdcall"]
extern "stdcall" {
    fn CommandLineToArgvW(lpCmdLine: LPCWSTR, pNumArgs: *mut c_int) -> **u16;
}

#[cfg(windows, target_arch = "x86_64")]
#[link_name="shell32"]
extern {
    fn CommandLineToArgvW(lpCmdLine: LPCWSTR, pNumArgs: *mut c_int) -> **u16;
}

struct OverriddenArgs {
    val: ~[~str]
}

/// Returns the arguments which this program was started with (normally passed
/// via the command line).
pub fn args() -> ~[~str] {
    real_args()
}

// FIXME #6100 we should really use an internal implementation of this - using
// the POSIX glob functions isn't portable to windows, probably has slight
// inconsistencies even where it is implemented, and makes extending
// functionality a lot more difficult
// FIXME #6101 also provide a non-allocating version - each_glob or so?
/// Returns a vector of Path objects that match the given glob pattern
#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "macos")]
pub fn glob(pattern: &str) -> ~[Path] {
    #[fixed_stack_segment]; #[inline(never)];

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    fn default_glob_t () -> libc::glob_t {
        libc::glob_t {
            gl_pathc: 0,
            gl_pathv: ptr::null(),
            gl_offs: 0,
            __unused1: ptr::null(),
            __unused2: ptr::null(),
            __unused3: ptr::null(),
            __unused4: ptr::null(),
            __unused5: ptr::null(),
        }
    }

    #[cfg(target_os = "freebsd")]
    fn default_glob_t () -> libc::glob_t {
        libc::glob_t {
            gl_pathc: 0,
            __unused1: 0,
            gl_offs: 0,
            __unused2: 0,
            gl_pathv: ptr::null(),
            __unused3: ptr::null(),
            __unused4: ptr::null(),
            __unused5: ptr::null(),
            __unused6: ptr::null(),
            __unused7: ptr::null(),
            __unused8: ptr::null(),
        }
    }

    #[cfg(target_os = "macos")]
    fn default_glob_t () -> libc::glob_t {
        libc::glob_t {
            gl_pathc: 0,
            __unused1: 0,
            gl_offs: 0,
            __unused2: 0,
            gl_pathv: ptr::null(),
            __unused3: ptr::null(),
            __unused4: ptr::null(),
            __unused5: ptr::null(),
            __unused6: ptr::null(),
            __unused7: ptr::null(),
            __unused8: ptr::null(),
        }
    }

    let mut g = default_glob_t();
    do pattern.with_c_str |c_pattern| {
        unsafe { libc::glob(c_pattern, 0, ptr::null(), &mut g) }
    };
    do(|| {
        let paths = unsafe {
            vec::raw::from_buf_raw(g.gl_pathv, g.gl_pathc as uint)
        };
        do paths.map |&c_str| {
            Path(unsafe { str::raw::from_c_str(c_str) })
        }
    }).finally {
        unsafe { libc::globfree(&mut g) };
    }
}

/// Returns a vector of Path objects that match the given glob pattern
#[cfg(target_os = "win32")]
pub fn glob(_pattern: &str) -> ~[Path] {
    fail!("glob() is unimplemented on Windows")
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

#[cfg(unix)]
pub fn page_size() -> uint {
    #[fixed_stack_segment]; #[inline(never)];

    unsafe {
        libc::sysconf(libc::_SC_PAGESIZE) as uint
    }
}

#[cfg(windows)]
pub fn page_size() -> uint {
    #[fixed_stack_segment]; #[inline(never)];

    unsafe {
        let mut info = libc::SYSTEM_INFO::new();
        libc::GetSystemInfo(&mut info);

        return info.dwPageSize as uint;
    }
}

pub struct MemoryMap {
    data: *mut u8,
    len: size_t,
    kind: MemoryMapKind
}

pub enum MemoryMapKind {
    MapFile(*c_void),
    MapVirtual
}

pub enum MapOption {
    MapReadable,
    MapWritable,
    MapExecutable,
    MapAddr(*c_void),
    MapFd(c_int),
    MapOffset(uint)
}

pub enum MapError {
    // Linux-specific errors
    ErrFdNotAvail,
    ErrInvalidFd,
    ErrUnaligned,
    ErrNoMapSupport,
    ErrNoMem,
    ErrUnknown(libc::c_int),

    // Windows-specific errors
    ErrUnsupProt,
    ErrUnsupOffset,
    ErrAlreadyExists,
    ErrVirtualAlloc(uint),
    ErrCreateFileMappingW(uint),
    ErrMapViewOfFile(uint)
}

impl to_str::ToStr for MapError {
    fn to_str(&self) -> ~str {
        match *self {
            ErrFdNotAvail => ~"fd not available for reading or writing",
            ErrInvalidFd => ~"Invalid fd",
            ErrUnaligned => ~"Unaligned address, invalid flags, \
                              negative length or unaligned offset",
            ErrNoMapSupport=> ~"File doesn't support mapping",
            ErrNoMem => ~"Invalid address, or not enough available memory",
            ErrUnknown(code) => fmt!("Unknown error=%?", code),
            ErrUnsupProt => ~"Protection mode unsupported",
            ErrUnsupOffset => ~"Offset in virtual memory mode is unsupported",
            ErrAlreadyExists => ~"File mapping for specified file already exists",
            ErrVirtualAlloc(code) => fmt!("VirtualAlloc failure=%?", code),
            ErrCreateFileMappingW(code) => fmt!("CreateFileMappingW failure=%?", code),
            ErrMapViewOfFile(code) => fmt!("MapViewOfFile failure=%?", code)
        }
    }
}

#[cfg(unix)]
impl MemoryMap {
    pub fn new(min_len: uint, options: ~[MapOption]) -> Result<~MemoryMap, MapError> {
        #[fixed_stack_segment]; #[inline(never)];

        use libc::off_t;

        let mut addr: *c_void = ptr::null();
        let mut prot: c_int = 0;
        let mut flags: c_int = libc::MAP_PRIVATE;
        let mut fd: c_int = -1;
        let mut offset: off_t = 0;
        let len = round_up(min_len, page_size()) as size_t;

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
                MapOffset(offset_) => { offset = offset_ as off_t; }
            }
        }
        if fd == -1 { flags |= libc::MAP_ANON; }

        let r = unsafe {
            libc::mmap(addr, len, prot, flags, fd, offset)
        };
        if r.equiv(&libc::MAP_FAILED) {
            Err(match errno() as c_int {
                libc::EACCES => ErrFdNotAvail,
                libc::EBADF => ErrInvalidFd,
                libc::EINVAL => ErrUnaligned,
                libc::ENODEV => ErrNoMapSupport,
                libc::ENOMEM => ErrNoMem,
                code => ErrUnknown(code)
            })
        } else {
            Ok(~MemoryMap {
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

    pub fn granularity() -> uint {
        page_size()
    }
}

#[cfg(unix)]
impl Drop for MemoryMap {
    fn drop(&self) {
        #[fixed_stack_segment]; #[inline(never)];

        unsafe {
            match libc::munmap(self.data as *c_void, self.len) {
                0 => (),
                -1 => error!(match errno() as c_int {
                    libc::EINVAL => ~"invalid addr or len",
                    e => fmt!("unknown errno=%?", e)
                }),
                r => error!(fmt!("Unexpected result %?", r))
            }
        }
    }
}

#[cfg(windows)]
impl MemoryMap {
    pub fn new(min_len: uint, options: ~[MapOption]) -> Result<~MemoryMap, MapError> {
        #[fixed_stack_segment]; #[inline(never)];

        use libc::types::os::arch::extra::{LPVOID, DWORD, SIZE_T, HANDLE};

        let mut lpAddress: LPVOID = ptr::mut_null();
        let mut readable = false;
        let mut writable = false;
        let mut executable = false;
        let mut fd: c_int = -1;
        let mut offset: uint = 0;
        let len = round_up(min_len, page_size()) as SIZE_T;

        for &o in options.iter() {
            match o {
                MapReadable => { readable = true; },
                MapWritable => { writable = true; },
                MapExecutable => { executable = true; }
                MapAddr(addr_) => { lpAddress = addr_ as LPVOID; },
                MapFd(fd_) => { fd = fd_; },
                MapOffset(offset_) => { offset = offset_; }
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
                                   len,
                                   libc::MEM_COMMIT | libc::MEM_RESERVE,
                                   flProtect)
            };
            match r as uint {
                0 => Err(ErrVirtualAlloc(errno())),
                _ => Ok(~MemoryMap {
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
                    _ => Ok(~MemoryMap {
                       data: r as *mut u8,
                       len: len,
                       kind: MapFile(mapping as *c_void)
                    })
                }
            }
        }
    }

    /// Granularity of MapAddr() and MapOffset() parameter values.
    /// This may be greater than the value returned by page_size().
    pub fn granularity() -> uint {
        #[fixed_stack_segment]; #[inline(never)];

        unsafe {
            let mut info = libc::SYSTEM_INFO::new();
            libc::GetSystemInfo(&mut info);

            return info.dwAllocationGranularity as uint;
        }
    }
}

#[cfg(windows)]
impl Drop for MemoryMap {
    fn drop(&self) {
        #[fixed_stack_segment]; #[inline(never)];

        use libc::types::os::arch::extra::{LPCVOID, HANDLE};
        use libc::consts::os::extra::FALSE;

        unsafe {
            match self.kind {
                MapVirtual => {
                    if libc::VirtualFree(self.data as *mut c_void,
                                         self.len,
                                         libc::MEM_RELEASE) == FALSE {
                        error!(fmt!("VirtualFree failed: %?", errno()));
                    }
                },
                MapFile(mapping) => {
                    if libc::UnmapViewOfFile(self.data as LPCVOID) == FALSE {
                        error!(fmt!("UnmapViewOfFile failed: %?", errno()));
                    }
                    if libc::CloseHandle(mapping as HANDLE) == FALSE {
                        error!(fmt!("CloseHandle failed: %?", errno()));
                    }
                }
            }
        }
    }
}

pub mod consts {

    #[cfg(unix)]
    pub use os::consts::unix::*;

    #[cfg(windows)]
    pub use os::consts::windows::*;

    #[cfg(target_os = "macos")]
    pub use os::consts::macos::*;

    #[cfg(target_os = "freebsd")]
    pub use os::consts::freebsd::*;

    #[cfg(target_os = "linux")]
    pub use os::consts::linux::*;

    #[cfg(target_os = "android")]
    pub use os::consts::android::*;

    #[cfg(target_os = "win32")]
    pub use os::consts::win32::*;

    #[cfg(target_arch = "x86")]
    pub use os::consts::x86::*;

    #[cfg(target_arch = "x86_64")]
    pub use os::consts::x86_64::*;

    #[cfg(target_arch = "arm")]
    pub use os::consts::arm::*;

    #[cfg(target_arch = "mips")]
    use os::consts::mips::*;

    pub mod unix {
        pub static FAMILY: &'static str = "unix";
    }

    pub mod windows {
        pub static FAMILY: &'static str = "windows";
    }

    pub mod macos {
        pub static SYSNAME: &'static str = "macos";
        pub static DLL_PREFIX: &'static str = "lib";
        pub static DLL_SUFFIX: &'static str = ".dylib";
        pub static EXE_SUFFIX: &'static str = "";
    }

    pub mod freebsd {
        pub static SYSNAME: &'static str = "freebsd";
        pub static DLL_PREFIX: &'static str = "lib";
        pub static DLL_SUFFIX: &'static str = ".so";
        pub static EXE_SUFFIX: &'static str = "";
    }

    pub mod linux {
        pub static SYSNAME: &'static str = "linux";
        pub static DLL_PREFIX: &'static str = "lib";
        pub static DLL_SUFFIX: &'static str = ".so";
        pub static EXE_SUFFIX: &'static str = "";
    }

    pub mod android {
        pub static SYSNAME: &'static str = "android";
        pub static DLL_PREFIX: &'static str = "lib";
        pub static DLL_SUFFIX: &'static str = ".so";
        pub static EXE_SUFFIX: &'static str = "";
    }

    pub mod win32 {
        pub static SYSNAME: &'static str = "win32";
        pub static DLL_PREFIX: &'static str = "";
        pub static DLL_SUFFIX: &'static str = ".dll";
        pub static EXE_SUFFIX: &'static str = ".exe";
    }


    pub mod x86 {
        pub static ARCH: &'static str = "x86";
    }
    pub mod x86_64 {
        pub static ARCH: &'static str = "x86_64";
    }
    pub mod arm {
        pub static ARCH: &'static str = "arm";
    }
    pub mod mips {
        pub static ARCH: &'static str = "mips";
    }
}

#[cfg(test)]
mod tests {
    use c_str::ToCStr;
    use libc::{c_int, c_void, size_t};
    use libc;
    use option::Some;
    use option;
    use os::{env, getcwd, getenv, make_absolute, args};
    use os::{remove_file, setenv, unsetenv};
    use os;
    use path::Path;
    use rand::RngUtil;
    use rand;
    use run;
    use str::StrSlice;
    use libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};


    #[test]
    pub fn last_os_error() {
        debug!(os::last_os_error());
    }

    #[test]
    pub fn test_args() {
        let a = args();
        assert!(a.len() >= 1);
    }

    fn make_rand_name() -> ~str {
        let mut rng = rand::rng();
        let n = ~"TEST" + rng.gen_str(10u);
        assert!(getenv(n).is_none());
        n
    }

    #[test]
    fn test_setenv() {
        let n = make_rand_name();
        setenv(n, "VALUE");
        assert_eq!(getenv(n), option::Some(~"VALUE"));
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
        assert_eq!(getenv(n), option::Some(~"2"));
        setenv(n, "");
        assert_eq!(getenv(n), option::Some(~""));
    }

    // Windows GetEnvironmentVariable requires some extra work to make sure
    // the buffer the variable is copied into is the right size
    #[test]
    #[ignore]
    fn test_getenv_big() {
        let mut s = ~"";
        let mut i = 0;
        while i < 100 {
            s = s + "aaaaaaaaaa";
            i += 1;
        }
        let n = make_rand_name();
        setenv(n, s);
        debug!(s.clone());
        assert_eq!(getenv(n), option::Some(s));
    }

    #[test]
    fn test_self_exe_path() {
        let path = os::self_exe_path();
        assert!(path.is_some());
        let path = path.unwrap();
        debug!(path.clone());

        // Hard to test this function
        assert!(path.is_absolute);
    }

    #[test]
    #[ignore]
    fn test_env_getenv() {
        let e = env();
        assert!(e.len() > 0u);
        for p in e.iter() {
            let (n, v) = (*p).clone();
            debug!(n.clone());
            let v2 = getenv(n);
            // MingW seems to set some funky environment variables like
            // "=C:=C:\MinGW\msys\1.0\bin" and "!::=::\" that are returned
            // from env() but not visible from getenv().
            assert!(v2.is_none() || v2 == option::Some(v));
        }
    }

    #[test]
    fn test_env_setenv() {
        let n = make_rand_name();

        let mut e = env();
        setenv(n, "VALUE");
        assert!(!e.contains(&(n.clone(), ~"VALUE")));

        e = env();
        assert!(e.contains(&(n, ~"VALUE")));
    }

    #[test]
    fn test() {
        assert!((!Path("test-path").is_absolute));

        debug!("Current working directory: %s", getcwd().to_str());

        debug!(make_absolute(&Path("test-path")));
        debug!(make_absolute(&Path("/usr/bin")));
    }

    #[test]
    #[cfg(unix)]
    fn homedir() {
        let oldhome = getenv("HOME");

        setenv("HOME", "/home/MountainView");
        assert_eq!(os::homedir(), Some(Path("/home/MountainView")));

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
        assert_eq!(os::homedir(), Some(Path("/home/MountainView")));

        setenv("HOME", "");

        setenv("USERPROFILE", "/home/MountainView");
        assert_eq!(os::homedir(), Some(Path("/home/MountainView")));

        setenv("HOME", "/home/MountainView");
        setenv("USERPROFILE", "/home/PaloAlto");
        assert_eq!(os::homedir(), Some(Path("/home/MountainView")));

        oldhome.iter().advance(|s| { setenv("HOME", *s); true });
        olduserprofile.iter().advance(|s| { setenv("USERPROFILE", *s); true });
    }

    #[test]
    fn tmpdir() {
        assert!(!os::tmpdir().to_str().is_empty());
    }

    // Issue #712
    #[test]
    fn test_list_dir_no_invalid_memory_access() {
        os::list_dir(&Path("."));
    }

    #[test]
    fn list_dir() {
        let dirs = os::list_dir(&Path("."));
        // Just assuming that we've got some contents in the current directory
        assert!(dirs.len() > 0u);

        for dir in dirs.iter() {
            debug!((*dir).clone());
        }
    }

    #[test]
    fn list_dir_empty_path() {
        let dirs = os::list_dir(&Path(""));
        assert!(dirs.is_empty());
    }

    #[test]
    #[cfg(not(windows))]
    fn list_dir_root() {
        let dirs = os::list_dir(&Path("/"));
        assert!(dirs.len() > 1);
    }
    #[test]
    #[cfg(windows)]
    fn list_dir_root() {
        let dirs = os::list_dir(&Path("C:\\"));
        assert!(dirs.len() > 1);
    }


    #[test]
    fn path_is_dir() {
        assert!((os::path_is_dir(&Path("."))));
        assert!((!os::path_is_dir(&Path("test/stdtest/fs.rs"))));
    }

    #[test]
    fn path_exists() {
        assert!((os::path_exists(&Path("."))));
        assert!((!os::path_exists(&Path(
                     "test/nonexistent-bogus-path"))));
    }

    #[test]
    fn copy_file_does_not_exist() {
      assert!(!os::copy_file(&Path("test/nonexistent-bogus-path"),
                            &Path("test/other-bogus-path")));
      assert!(!os::path_exists(&Path("test/other-bogus-path")));
    }

    #[test]
    fn copy_file_ok() {
        #[fixed_stack_segment]; #[inline(never)];

        unsafe {
            let tempdir = getcwd(); // would like to use $TMPDIR,
                                    // doesn't seem to work on Linux
            assert!((tempdir.to_str().len() > 0u));
            let input = tempdir.push("in.txt");
            let out = tempdir.push("out.txt");

            /* Write the temp input file */
            let ostream = do input.with_c_str |fromp| {
                do "w+b".with_c_str |modebuf| {
                    libc::fopen(fromp, modebuf)
                }
            };
            assert!((ostream as uint != 0u));
            let s = ~"hello";
            do "hello".with_c_str |buf| {
                let write_len = libc::fwrite(buf as *c_void,
                                             1u as size_t,
                                             (s.len() + 1u) as size_t,
                                             ostream);
                assert_eq!(write_len, (s.len() + 1) as size_t)
            }
            assert_eq!(libc::fclose(ostream), (0u as c_int));
            let in_mode = input.get_mode();
            let rs = os::copy_file(&input, &out);
            if (!os::path_exists(&input)) {
                fail!("%s doesn't exist", input.to_str());
            }
            assert!((rs));
            let rslt = run::process_status("diff", [input.to_str(), out.to_str()]);
            assert_eq!(rslt, 0);
            assert_eq!(out.get_mode(), in_mode);
            assert!((remove_file(&input)));
            assert!((remove_file(&out)));
        }
    }

    #[test]
    fn recursive_mkdir_slash() {
        let path = Path("/");
        assert!(os::mkdir_recursive(&path,  (S_IRUSR | S_IWUSR | S_IXUSR) as i32));
    }

    #[test]
    fn recursive_mkdir_empty() {
        let path = Path("");
        assert!(!os::mkdir_recursive(&path, (S_IRUSR | S_IWUSR | S_IXUSR) as i32));
    }

    #[test]
    fn memory_map_rw() {
        use result::{Ok, Err};

        let chunk = match os::MemoryMap::new(16, ~[
            os::MapReadable,
            os::MapWritable
        ]) {
            Ok(chunk) => chunk,
            Err(msg) => fail!(msg.to_str())
        };
        assert!(chunk.len >= 16);

        unsafe {
            *chunk.data = 0xBE;
            assert!(*chunk.data == 0xBE);
        }
    }

    #[test]
    fn memory_map_file() {
        #[fixed_stack_segment]; #[inline(never)];

        use result::{Ok, Err};
        use os::*;
        use libc::*;

        #[cfg(unix)]
        #[fixed_stack_segment]
        #[inline(never)]
        fn lseek_(fd: c_int, size: uint) {
            unsafe {
                assert!(lseek(fd, size as off_t, SEEK_SET) == size as off_t);
            }
        }
        #[cfg(windows)]
        #[fixed_stack_segment]
        #[inline(never)]
        fn lseek_(fd: c_int, size: uint) {
           unsafe {
               assert!(lseek(fd, size as c_long, SEEK_SET) == size as c_long);
           }
        }

        let path = tmpdir().push("mmap_file.tmp");
        let size = MemoryMap::granularity() * 2;
        remove_file(&path);

        let fd = unsafe {
            let fd = do path.with_c_str |path| {
                open(path, O_CREAT | O_RDWR | O_TRUNC, S_IRUSR | S_IWUSR)
            };
            lseek_(fd, size);
            do "x".with_c_str |x| {
                assert!(write(fd, x as *c_void, 1) == 1);
            }
            fd
        };
        let chunk = match MemoryMap::new(size / 2, ~[
            MapReadable,
            MapWritable,
            MapFd(fd),
            MapOffset(size / 2)
        ]) {
            Ok(chunk) => chunk,
            Err(msg) => fail!(msg.to_str())
        };
        assert!(chunk.len > 0);

        unsafe {
            *chunk.data = 0xbe;
            assert!(*chunk.data == 0xbe);
            close(fd);
        }
    }

    // More recursive_mkdir tests are in extra::tempfile
}
