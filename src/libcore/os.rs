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

import libc::{c_char, c_void, c_int, c_uint, size_t, ssize_t,
              mode_t, pid_t, FILE};
import libc::{close, fclose};

import option::{some, none};

import getcwd = rustrt::rust_getcwd;
import consts::*;

export close, fclose, fsync_fd, waitpid;
export env, getenv, setenv, fdopen, pipe;
export getcwd, dll_filename, self_exe_path;
export exe_suffix, dll_suffix, sysname, arch, family;
export homedir, list_dir, list_dir_path, path_is_dir, path_exists,
       make_absolute, make_dir, remove_dir, change_dir, remove_file,
       copy_file;
export last_os_error;
export set_exit_status;
export walk_dir;

// FIXME: move these to str perhaps? #2620
export as_c_charp, fill_charp_buf;

extern mod rustrt {
    fn rust_env_pairs() -> ~[str];
    fn rust_getcwd() -> str;
    fn rust_path_is_dir(path: *libc::c_char) -> c_int;
    fn rust_path_exists(path: *libc::c_char) -> c_int;
    fn rust_list_files(path: str) -> ~[str];
    fn rust_process_wait(handle: c_int) -> c_int;
    fn last_os_error() -> str;
    fn rust_set_exit_status(code: libc::intptr_t);
}


fn env() -> ~[(str,str)] {
    let mut pairs = ~[];
    for vec::each(rustrt::rust_env_pairs()) |p| {
        let vs = str::splitn_char(p, '=', 1u);
        assert vec::len(vs) == 2u;
        vec::push(pairs, (vs[0], vs[1]));
    }
    ret pairs;
}

const tmpbuf_sz : uint = 1000u;

fn as_c_charp<T>(s: str, f: fn(*c_char) -> T) -> T {
    str::as_c_str(s, |b| f(b as *c_char))
}

fn fill_charp_buf(f: fn(*mut c_char, size_t) -> bool)
    -> option<str> {
    let buf = vec::to_mut(vec::from_elem(tmpbuf_sz, 0u8 as c_char));
    do vec::as_mut_buf(buf) |b| {
        if f(b, tmpbuf_sz as size_t) unsafe {
            some(str::unsafe::from_buf(b as *u8))
        } else {
            none
        }
    }
}

#[cfg(windows)]
mod win32 {
    import dword = libc::types::os::arch::extra::DWORD;

    fn fill_utf16_buf_and_decode(f: fn(*mut u16, dword) -> dword)
        -> option<str> {

        // FIXME: remove these when export globs work properly. #1238
        import libc::funcs::extra::kernel32::*;
        import libc::consts::os::extra::*;

        let mut n = tmpbuf_sz as dword;
        let mut res = none;
        let mut done = false;
        while !done {
            let buf = vec::to_mut(vec::from_elem(n as uint, 0u16));
            do vec::as_mut_buf(buf) |b| {
                let k : dword = f(b, tmpbuf_sz as dword);
                if k == (0 as dword) {
                    done = true;
                } else if (k == n &&
                           GetLastError() ==
                           ERROR_INSUFFICIENT_BUFFER as dword) {
                    n *= (2 as dword);
                } else {
                    let sub = vec::slice(buf, 0u, k as uint);
                    res = option::some::<str>(str::from_utf16(sub));
                    done = true;
                }
            }
        }
        ret res;
    }

    fn as_utf16_p<T>(s: str, f: fn(*u16) -> T) -> T {
        let mut t = str::to_utf16(s);
        // Null terminate before passing on.
        t += ~[0u16];
        vec::as_buf(t, f)
    }
}

fn getenv(n: str) -> option<str> {
    global_env::getenv(n)
}

fn setenv(n: str, v: str) {
    global_env::setenv(n, v)
}

mod global_env {
    //! Internal module for serializing access to getenv/setenv

    export getenv;
    export setenv;

    extern mod rustrt {
        fn rust_global_env_chan_ptr() -> *libc::uintptr_t;
    }

    enum msg {
        msg_getenv(str, comm::chan<option<str>>),
        msg_setenv(str, str, comm::chan<()>)
    }

    fn getenv(n: str) -> option<str> {
        let env_ch = get_global_env_chan();
        let po = comm::port();
        comm::send(env_ch, msg_getenv(n, comm::chan(po)));
        comm::recv(po)
    }

    fn setenv(n: str, v: str) {
        let env_ch = get_global_env_chan();
        let po = comm::port();
        comm::send(env_ch, msg_setenv(n, v, comm::chan(po)));
        comm::recv(po)
    }

    fn get_global_env_chan() -> comm::chan<msg> {
        let global_ptr = rustrt::rust_global_env_chan_ptr();
        let builder_fn = || {
            let builder = task::builder();
            task::unsupervise(builder);
            task::set_opts(builder, {
                sched:  some({
                    mode: task::single_threaded,
                    // FIXME (#2621): This would be a good place to use a
                    // very small foreign stack
                    foreign_stack_size: none
                })
                with task::get_opts(builder)
            });
            builder
        };
        unsafe {
            priv::chan_from_global_ptr(
                global_ptr, builder_fn, global_env_task)
        }
    }

    fn global_env_task(msg_po: comm::port<msg>) {
        unsafe {
            do priv::weaken_task |weak_po| {
                loop {
                    alt comm::select2(msg_po, weak_po) {
                      either::left(msg_getenv(n, resp_ch)) {
                        comm::send(resp_ch, impl::getenv(n))
                      }
                      either::left(msg_setenv(n, v, resp_ch)) {
                        comm::send(resp_ch, impl::setenv(n, v))
                      }
                      either::right(_) {
                        break;
                      }
                    }
                }
            }
        }
    }

    mod impl {

        #[cfg(unix)]
        fn getenv(n: str) -> option<str> {
            unsafe {
                let s = str::as_c_str(n, libc::getenv);
                ret if unsafe::reinterpret_cast(s) == 0 {
                    option::none::<str>
                } else {
                    let s = unsafe::reinterpret_cast(s);
                    option::some::<str>(str::unsafe::from_buf(s))
                };
            }
        }

        #[cfg(windows)]
        fn getenv(n: str) -> option<str> {
            import libc::types::os::arch::extra::*;
            import libc::funcs::extra::kernel32::*;
            import win32::*;
            do as_utf16_p(n) |u| {
                do fill_utf16_buf_and_decode() |buf, sz| {
                    GetEnvironmentVariableW(u, buf, sz)
                }
            }
        }


        #[cfg(unix)]
        fn setenv(n: str, v: str) {

            // FIXME: remove this when export globs work properly. #1238
            import libc::funcs::posix01::unistd::setenv;
            do str::as_c_str(n) |nbuf| {
                do str::as_c_str(v) |vbuf| {
                    setenv(nbuf, vbuf, 1i32);
                }
            }
        }


        #[cfg(windows)]
        fn setenv(n: str, v: str) {
            // FIXME: remove imports when export globs work properly. #1238
            import libc::funcs::extra::kernel32::*;
            import win32::*;
            do as_utf16_p(n) |nbuf| {
                do as_utf16_p(v) |vbuf| {
                    SetEnvironmentVariableW(nbuf, vbuf);
                }
            }
        }

    }
}

fn fdopen(fd: c_int) -> *FILE {
    ret do as_c_charp("r") |modebuf| {
        libc::fdopen(fd, modebuf)
    };
}


// fsync related

#[cfg(windows)]
fn fsync_fd(fd: c_int, _level: io::fsync::level) -> c_int {
    import libc::funcs::extra::msvcrt::*;
    ret commit(fd);
}

#[cfg(target_os = "linux")]
fn fsync_fd(fd: c_int, level: io::fsync::level) -> c_int {
    import libc::funcs::posix01::unistd::*;
    alt level {
      io::fsync::fsync
      | io::fsync::fullfsync { ret fsync(fd); }
      io::fsync::fdatasync { ret fdatasync(fd); }
    }
}

#[cfg(target_os = "macos")]
fn fsync_fd(fd: c_int, level: io::fsync::level) -> c_int {
    import libc::consts::os::extra::*;
    import libc::funcs::posix88::fcntl::*;
    import libc::funcs::posix01::unistd::*;
    alt level {
      io::fsync::fsync { ret fsync(fd); }
      _ {
        // According to man fnctl, the ok retval is only specified to be !=-1
        if (fcntl(F_FULLFSYNC as c_int, fd) == -1 as c_int)
            { ret -1 as c_int; }
        else
            { ret 0 as c_int; }
      }
    }
}

#[cfg(target_os = "freebsd")]
fn fsync_fd(fd: c_int, _l: io::fsync::level) -> c_int {
    import libc::funcs::posix01::unistd::*;
    ret fsync(fd);
}


#[cfg(windows)]
fn waitpid(pid: pid_t) -> c_int {
    ret rustrt::rust_process_wait(pid);
}

#[cfg(unix)]
fn waitpid(pid: pid_t) -> c_int {
    import libc::funcs::posix01::wait::*;
    let status = 0 as c_int;

    assert (waitpid(pid, ptr::mut_addr_of(status),
                    0 as c_int) != (-1 as c_int));
    ret status;
}


#[cfg(unix)]
fn pipe() -> {in: c_int, out: c_int} {
    let fds = {mut in: 0 as c_int,
               mut out: 0 as c_int };
    assert (libc::pipe(ptr::mut_addr_of(fds.in)) == (0 as c_int));
    ret {in: fds.in, out: fds.out};
}



#[cfg(windows)]
fn pipe() -> {in: c_int, out: c_int} {
    // FIXME: remove this when export globs work properly. #1238
    import libc::consts::os::extra::*;
    // Windows pipes work subtly differently than unix pipes, and their
    // inheritance has to be handled in a different way that I do not fully
    // understand. Here we explicitly make the pipe non-inheritable, which
    // means to pass it to a subprocess they need to be duplicated first, as
    // in rust_run_program.
    let fds = { mut in: 0 as c_int,
               mut out: 0 as c_int };
    let res = libc::pipe(ptr::mut_addr_of(fds.in),
                         1024 as c_uint,
                         (O_BINARY | O_NOINHERIT) as c_int);
    assert (res == 0 as c_int);
    assert (fds.in != -1 as c_int && fds.in != 0 as c_int);
    assert (fds.out != -1 as c_int && fds.in != 0 as c_int);
    ret {in: fds.in, out: fds.out};
}


fn dll_filename(base: str) -> str {
    ret pre() + base + dll_suffix();

    #[cfg(unix)]
    fn pre() -> str { "lib" }

    #[cfg(windows)]
    fn pre() -> str { "" }
}


fn self_exe_path() -> option<path> {

    #[cfg(target_os = "freebsd")]
    fn load_self() -> option<path> {
        unsafe {
            import libc::funcs::bsd44::*;
            import libc::consts::os::extra::*;
            do fill_charp_buf() |buf, sz| {
                let mib = ~[CTL_KERN as c_int,
                           KERN_PROC as c_int,
                           KERN_PROC_PATHNAME as c_int, -1 as c_int];
                sysctl(vec::unsafe::to_ptr(mib), vec::len(mib) as c_uint,
                       buf as *mut c_void, ptr::mut_addr_of(sz),
                       ptr::null(), 0u as size_t) == (0 as c_int)
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn load_self() -> option<path> {
        import libc::funcs::posix01::unistd::readlink;
        do fill_charp_buf() |buf, sz| {
            do as_c_charp("/proc/self/exe") |proc_self_buf| {
                readlink(proc_self_buf, buf, sz) != (-1 as ssize_t)
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn load_self() -> option<path> {
        // FIXME: remove imports when export globs work properly. #1238
        import libc::funcs::extra::*;
        do fill_charp_buf() |buf, sz| {
            _NSGetExecutablePath(buf, ptr::mut_addr_of(sz as u32))
                == (0 as c_int)
        }
    }

    #[cfg(windows)]
    fn load_self() -> option<path> {
        // FIXME: remove imports when export globs work properly. #1238
        import libc::types::os::arch::extra::*;
        import libc::funcs::extra::kernel32::*;
        import win32::*;
        do fill_utf16_buf_and_decode() |buf, sz| {
            GetModuleFileNameW(0u as dword, buf, sz)
        }
    }

    do option::map(load_self()) |pth| {
        path::dirname(pth) + path::path_sep()
    }
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
fn homedir() -> option<path> {
    ret alt getenv("HOME") {
        some(p) {
            if !str::is_empty(p) {
                some(p)
            } else {
                secondary()
            }
        }
        none {
            secondary()
        }
    };

    #[cfg(unix)]
    fn secondary() -> option<path> {
        none
    }

    #[cfg(windows)]
    fn secondary() -> option<path> {
        do option::chain(getenv("USERPROFILE")) |p| {
            if !str::is_empty(p) {
                some(p)
            } else {
                none
            }
        }
    }
}

/// Recursively walk a directory structure
fn walk_dir(p: path, f: fn(path) -> bool) {

    walk_dir_(p, f);

    fn walk_dir_(p: path, f: fn(path) -> bool) -> bool {
        let mut keepgoing = true;
        do list_dir(p).each |q| {
            let path = path::connect(p, q);
            if !f(path) {
                keepgoing = false;
                false
            } else {
                if path_is_dir(path) {
                    if !walk_dir_(path, f) {
                        keepgoing = false;
                        false
                    } else {
                        true
                    }
                } else {
                    true
                }
            }
        }
        ret keepgoing;
    }
}

/// Indicates whether a path represents a directory
fn path_is_dir(p: path) -> bool {
    do str::as_c_str(p) |buf| {
        rustrt::rust_path_is_dir(buf) != 0 as c_int
    }
}

/// Indicates whether a path exists
fn path_exists(p: path) -> bool {
    do str::as_c_str(p) |buf| {
        rustrt::rust_path_exists(buf) != 0 as c_int
    }
}

// FIXME (#2622): under Windows, we should prepend the current drive letter
// to paths that start with a slash.
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
fn make_absolute(p: path) -> path {
    if path::path_is_absolute(p) {
        p
    } else {
        path::connect(getcwd(), p)
    }
}


/// Creates a directory at the specified path
fn make_dir(p: path, mode: c_int) -> bool {
    ret mkdir(p, mode);

    #[cfg(windows)]
    fn mkdir(p: path, _mode: c_int) -> bool {
        // FIXME: remove imports when export globs work properly. #1238
        import libc::types::os::arch::extra::*;
        import libc::funcs::extra::kernel32::*;
        import win32::*;
        // FIXME: turn mode into something useful? #2623
        do as_utf16_p(p) |buf| {
            CreateDirectoryW(buf, unsafe { unsafe::reinterpret_cast(0) })
                != (0 as BOOL)
        }
    }

    #[cfg(unix)]
    fn mkdir(p: path, mode: c_int) -> bool {
        do as_c_charp(p) |c| {
            libc::mkdir(c, mode as mode_t) == (0 as c_int)
        }
    }
}

/// Lists the contents of a directory
fn list_dir(p: path) -> ~[str] {

    #[cfg(unix)]
    fn star(p: str) -> str { p }

    #[cfg(windows)]
    fn star(p: str) -> str {
        let pl = str::len(p);
        if pl == 0u || (p[pl - 1u] as char != path::consts::path_sep
                        || p[pl - 1u] as char != path::consts::alt_path_sep) {
            p + path::path_sep() + "*"
        } else {
            p + "*"
        }
    }

    do rustrt::rust_list_files(star(p)).filter |filename| {
        !str::eq(filename, ".") && !str::eq(filename, "..")
    }
}

/**
 * Lists the contents of a directory
 *
 * This version prepends each entry with the directory.
 */
fn list_dir_path(p: path) -> ~[str] {
    let mut p = p;
    let pl = str::len(p);
    if pl == 0u || (p[pl - 1u] as char != path::consts::path_sep
                    && p[pl - 1u] as char != path::consts::alt_path_sep) {
        p += path::path_sep();
    }
    os::list_dir(p).map(|f| p + f)
}

/// Removes a directory at the specified path
fn remove_dir(p: path) -> bool {
   ret rmdir(p);

    #[cfg(windows)]
    fn rmdir(p: path) -> bool {
        // FIXME: remove imports when export globs work properly. #1238
        import libc::funcs::extra::kernel32::*;
        import libc::types::os::arch::extra::*;
        import win32::*;
        ret do as_utf16_p(p) |buf| {
            RemoveDirectoryW(buf) != (0 as BOOL)
        };
    }

    #[cfg(unix)]
    fn rmdir(p: path) -> bool {
        ret do as_c_charp(p) |buf| {
            libc::rmdir(buf) == (0 as c_int)
        };
    }
}

fn change_dir(p: path) -> bool {
    ret chdir(p);

    #[cfg(windows)]
    fn chdir(p: path) -> bool {
        // FIXME: remove imports when export globs work properly. #1238
        import libc::funcs::extra::kernel32::*;
        import libc::types::os::arch::extra::*;
        import win32::*;
        ret do as_utf16_p(p) |buf| {
            SetCurrentDirectoryW(buf) != (0 as BOOL)
        };
    }

    #[cfg(unix)]
    fn chdir(p: path) -> bool {
        ret do as_c_charp(p) |buf| {
            libc::chdir(buf) == (0 as c_int)
        };
    }
}

/// Copies a file from one location to another
fn copy_file(from: path, to: path) -> bool {
    ret do_copy_file(from, to);

    #[cfg(windows)]
    fn do_copy_file(from: path, to: path) -> bool {
        // FIXME: remove imports when export globs work properly. #1238
        import libc::funcs::extra::kernel32::*;
        import libc::types::os::arch::extra::*;
        import win32::*;
        ret do as_utf16_p(from) |fromp| {
            do as_utf16_p(to) |top| {
                CopyFileW(fromp, top, (0 as BOOL)) != (0 as BOOL)
            }
        }
    }

    #[cfg(unix)]
    fn do_copy_file(from: path, to: path) -> bool {
        let istream = do as_c_charp(from) |fromp| {
            do as_c_charp("rb") |modebuf| {
                libc::fopen(fromp, modebuf)
            }
        };
        if istream as uint == 0u {
            ret false;
        }
        let ostream = do as_c_charp(to) |top| {
            do as_c_charp("w+b") |modebuf| {
                libc::fopen(top, modebuf)
            }
        };
        if ostream as uint == 0u {
            fclose(istream);
            ret false;
        }
        let mut buf : ~[mut u8] = ~[mut];
        let bufsize = 8192u;
        vec::reserve(buf, bufsize);
        let mut done = false;
        let mut ok = true;
        while !done {
            do vec::as_mut_buf(buf) |b| {
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
        ret ok;
    }
}

/// Deletes an existing file
fn remove_file(p: path) -> bool {
    ret unlink(p);

    #[cfg(windows)]
    fn unlink(p: path) -> bool {
        // FIXME (similar to Issue #2006): remove imports when export globs
        // work properly.
        import libc::funcs::extra::kernel32::*;
        import libc::types::os::arch::extra::*;
        import win32::*;
        ret do as_utf16_p(p) |buf| {
            DeleteFileW(buf) != (0 as BOOL)
        };
    }

    #[cfg(unix)]
    fn unlink(p: path) -> bool {
        ret do as_c_charp(p) |buf| {
            libc::unlink(buf) == (0 as c_int)
        };
    }
}

/// Get a string representing the platform-dependent last error
fn last_os_error() -> str {
    rustrt::last_os_error()
}

/**
 * Sets the process exit code
 *
 * Sets the exit code returned by the process if all supervised tasks
 * terminate successfully (without failing). If the current root task fails
 * and is supervised by the scheduler then any user-specified exit status is
 * ignored and the process exits with the default failure status
 */
fn set_exit_status(code: int) {
    rustrt::rust_set_exit_status(code as libc::intptr_t);
}

#[cfg(unix)]
fn family() -> str { "unix" }

#[cfg(windows)]
fn family() -> str { "windows" }

#[cfg(target_os = "macos")]
mod consts {
    fn sysname() -> str { "macos" }
    fn exe_suffix() -> str { "" }
    fn dll_suffix() -> str { ".dylib" }
}

#[cfg(target_os = "freebsd")]
mod consts {
    fn sysname() -> str { "freebsd" }
    fn exe_suffix() -> str { "" }
    fn dll_suffix() -> str { ".so" }
}

#[cfg(target_os = "linux")]
mod consts {
    fn sysname() -> str { "linux" }
    fn exe_suffix() -> str { "" }
    fn dll_suffix() -> str { ".so" }
}

#[cfg(target_os = "win32")]
mod consts {
    fn sysname() -> str { "win32" }
    fn exe_suffix() -> str { ".exe" }
    fn dll_suffix() -> str { ".dll" }
}

#[cfg(target_arch = "x86")]
fn arch() -> str { "x86" }

#[cfg(target_arch = "x86_64")]
fn arch() -> str { "x86_64" }

#[cfg(target_arch = "arm")]
fn arch() -> str { "arm" }

#[cfg(test)]
mod tests {

    #[test]
    fn last_os_error() {
        log(debug, last_os_error());
    }

    fn make_rand_name() -> str {
        import rand;
        let rng: rand::rng = rand::rng();
        let n = "TEST" + rng.gen_str(10u);
        assert option::is_none(getenv(n));
        n
    }

    #[test]
    fn test_setenv() {
        let n = make_rand_name();
        setenv(n, "VALUE");
        assert getenv(n) == option::some("VALUE");
    }

    #[test]
    #[ignore(cfg(windows))]
    #[ignore]
    fn test_setenv_overwrite() {
        let n = make_rand_name();
        setenv(n, "1");
        setenv(n, "2");
        assert getenv(n) == option::some("2");
        setenv(n, "");
        assert getenv(n) == option::some("");
    }

    // Windows GetEnvironmentVariable requires some extra work to make sure
    // the buffer the variable is copied into is the right size
    #[test]
    #[ignore(cfg(windows))]
    #[ignore]
    fn test_getenv_big() {
        let mut s = "";
        let mut i = 0;
        while i < 100 { s += "aaaaaaaaaa"; i += 1; }
        let n = make_rand_name();
        setenv(n, s);
        log(debug, s);
        assert getenv(n) == option::some(s);
    }

    #[test]
    fn test_self_exe_path() {
        let path = os::self_exe_path();
        assert option::is_some(path);
        let path = option::get(path);
        log(debug, path);

        // Hard to test this function
        if os::sysname() != "win32" {
            assert str::starts_with(path, path::path_sep());
        } else {
            assert path[1] == ':' as u8;
        }
    }

    #[test]
    #[ignore]
    fn test_env_getenv() {
        let e = env();
        assert vec::len(e) > 0u;
        for vec::each(e) |p| {
            let (n, v) = p;
            log(debug, n);
            let v2 = getenv(n);
            // MingW seems to set some funky environment variables like
            // "=C:=C:\MinGW\msys\1.0\bin" and "!::=::\" that are returned
            // from env() but not visible from getenv().
            assert option::is_none(v2) || v2 == option::some(v);
        }
    }

    #[test]
    fn test_env_setenv() {
        let n = make_rand_name();

        let mut e = env();
        setenv(n, "VALUE");
        assert !vec::contains(e, (n, "VALUE"));

        e = env();
        assert vec::contains(e, (n, "VALUE"));
    }

    #[test]
    fn test() {
        assert (!path::path_is_absolute("test-path"));

        log(debug, "Current working directory: " + getcwd());

        log(debug, make_absolute("test-path"));
        log(debug, make_absolute("/usr/bin"));
    }

    #[test]
    #[cfg(unix)]
    fn homedir() {
        let oldhome = getenv("HOME");

        setenv("HOME", "/home/MountainView");
        assert os::homedir() == some("/home/MountainView");

        setenv("HOME", "");
        assert os::homedir() == none;

        option::iter(oldhome, |s| setenv("HOME", s));
    }

    #[test]
    #[cfg(windows)]
    fn homedir() {

        let oldhome = getenv("HOME");
        let olduserprofile = getenv("USERPROFILE");

        setenv("HOME", "");
        setenv("USERPROFILE", "");

        assert os::homedir() == none;

        setenv("HOME", "/home/MountainView");
        assert os::homedir() == some("/home/MountainView");

        setenv("HOME", "");

        setenv("USERPROFILE", "/home/MountainView");
        assert os::homedir() == some("/home/MountainView");

        setenv("USERPROFILE", "/home/MountainView");
        assert os::homedir() == some("/home/MountainView");

        setenv("HOME", "/home/MountainView");
        setenv("USERPROFILE", "/home/PaloAlto");
        assert os::homedir() == some("/home/MountainView");

        option::iter(oldhome, |s| setenv("HOME", s));
        option::iter(olduserprofile,
                               |s| setenv("USERPROFILE", s));
    }

    // Issue #712
    #[test]
    fn test_list_dir_no_invalid_memory_access() { os::list_dir("."); }

    #[test]
    fn list_dir() {
        let dirs = os::list_dir(".");
        // Just assuming that we've got some contents in the current directory
        assert (vec::len(dirs) > 0u);

        for vec::each(dirs) |dir| { log(debug, dir); }
    }

    #[test]
    fn path_is_dir() {
        assert (os::path_is_dir("."));
        assert (!os::path_is_dir("test/stdtest/fs.rs"));
    }

    #[test]
    fn path_exists() {
        assert (os::path_exists("."));
        assert (!os::path_exists("test/nonexistent-bogus-path"));
    }

    #[test]
    fn copy_file_does_not_exist() {
      assert !os::copy_file("test/nonexistent-bogus-path",
                           "test/other-bogus-path");
      assert !os::path_exists("test/other-bogus-path");
    }

    #[test]
    fn copy_file_ok() {
      let tempdir = getcwd(); // would like to use $TMPDIR,
                              // doesn't seem to work on Linux
      assert (str::len(tempdir) > 0u);
      let in = tempdir + path::path_sep() + "in.txt";
      let out = tempdir + path::path_sep() + "out.txt";

      /* Write the temp input file */
        let ostream = do as_c_charp(in) |fromp| {
            do as_c_charp("w+b") |modebuf| {
                libc::fopen(fromp, modebuf)
            }
      };
      assert (ostream as uint != 0u);
      let s = "hello";
      let mut buf = vec::to_mut(str::bytes(s) + ~[0 as u8]);
      do vec::as_mut_buf(buf) |b| {
          assert (libc::fwrite(b as *c_void, 1u as size_t,
                               (str::len(s) + 1u) as size_t, ostream)
                  == buf.len() as size_t)};
      assert (libc::fclose(ostream) == (0u as c_int));
      let rs = os::copy_file(in, out);
      if (!os::path_exists(in)) {
        fail (#fmt("%s doesn't exist", in));
      }
      assert(rs);
      let rslt = run::run_program("diff", ~[in, out]);
      assert (rslt == 0);
      assert (remove_file(in));
      assert (remove_file(out));
    }
}
