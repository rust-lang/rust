#[doc = "
Higher-level interfaces to libc::* functions and operating system services.

In general these take and return rust types, use rust idioms (enums,
closures, vectors) rather than C idioms, and do more extensive safety
checks.

This module is not meant to only contain 1:1 mappings to libc entries; any
os-interface code that is reasonably useful and broadly applicable can go
here. Including utility routines that merely build on other os code.

We assume the general case is that users do not care, and do not want to
be made to care, which operating system they are on. While they may want
to special case various special cases -- and so we will not _hide_ the
facts of which OS the user is on -- they should be given the opportunity
to write OS-ignorant code by default.
"];

import libc::{c_char, c_void, c_int, c_uint, size_t, ssize_t,
              mode_t, pid_t, FILE};
import libc::{close, fclose};

import option::{some, none};

import getcwd = rustrt::rust_getcwd;
import consts::*;

export close, fclose, fsync_fd, waitpid;
export env, getenv, setenv, fdopen, pipe;
export getcwd, dll_filename, self_exe_path;
export exe_suffix, dll_suffix, sysname;
export homedir, list_dir, list_dir_path, path_is_dir, path_exists,
       make_absolute, make_dir, remove_dir, change_dir, remove_file;

// FIXME: move these to str perhaps?
export as_c_charp, fill_charp_buf;

native mod rustrt {
    fn rust_env_pairs() -> [str];
    fn rust_getcwd() -> str;
    fn rust_path_is_dir(path: *libc::c_char) -> c_int;
    fn rust_path_exists(path: *libc::c_char) -> c_int;
    fn rust_list_files(path: str) -> [str];
    fn rust_process_wait(handle: c_int) -> c_int;
}


fn env() -> [(str,str)] {
    let mut pairs = [];
    for vec::each(rustrt::rust_env_pairs()) {|p|
        let vs = str::splitn_char(p, '=', 1u);
        assert vec::len(vs) == 2u;
        pairs += [(vs[0], vs[1])];
    }
    ret pairs;
}

const tmpbuf_sz : uint = 1000u;

fn as_c_charp<T>(s: str, f: fn(*c_char) -> T) -> T {
    str::as_c_str(s) {|b| f(b as *c_char) }
}

fn fill_charp_buf(f: fn(*mut c_char, size_t) -> bool)
    -> option<str> {
    let buf = vec::to_mut(vec::from_elem(tmpbuf_sz, 0u8 as c_char));
    vec::as_mut_buf(buf) { |b|
        if f(b, tmpbuf_sz as size_t) unsafe {
            some(str::unsafe::from_buf(b as *u8))
        } else {
            none
        }
    }
}

#[cfg(target_os = "win32")]
mod win32 {
    import dword = libc::types::os::arch::extra::DWORD;

    fn fill_utf16_buf_and_decode(f: fn(*mut u16, dword) -> dword)
        -> option<str> {

        // FIXME: remove these when export globs work properly.
        import libc::funcs::extra::kernel32::*;
        import libc::consts::os::extra::*;

        let mut n = tmpbuf_sz;
        let mut res = none;
        let mut done = false;
        while !done {
            let buf = vec::to_mut(vec::from_elem(n, 0u16));
            vec::as_mut_buf(buf) {|b|
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
        t += [0u16];
        vec::as_buf(t, f)
    }
}


#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
fn getenv(n: str) -> option<str> unsafe {
    let s = as_c_charp(n, libc::getenv);
    ret if unsafe::reinterpret_cast(s) == 0 {
            option::none::<str>
        } else {
            let s = unsafe::reinterpret_cast(s);
            option::some::<str>(str::unsafe::from_buf(s))
        };
}

#[cfg(target_os = "win32")]
fn getenv(n: str) -> option<str> unsafe {
    import libc::types::os::arch::extra::*;
    import libc::funcs::extra::kernel32::*;
    import win32::*;
    as_utf16_p(n) {|u|
        fill_utf16_buf_and_decode() {|buf, sz|
            GetEnvironmentVariableW(u, buf, sz)
        }
    }
}


#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
fn setenv(n: str, v: str) {

    // FIXME: remove this when export globs work properly.
    import libc::funcs::posix01::unistd::setenv;
    as_c_charp(n) {|nbuf|
        as_c_charp(v) {|vbuf|
            setenv(nbuf, vbuf, 1i32);
        }
    }
}


#[cfg(target_os = "win32")]
fn setenv(n: str, v: str) {
    // FIXME: remove imports when export globs work properly.
    import libc::funcs::extra::kernel32::*;
    import win32::*;
    as_utf16_p(n) {|nbuf|
        as_utf16_p(v) {|vbuf|
            SetEnvironmentVariableW(nbuf, vbuf);
        }
    }
}


fn fdopen(fd: c_int) -> *FILE {
    ret as_c_charp("r") {|modebuf|
        libc::fdopen(fd, modebuf)
    };
}


// fsync related

#[cfg(target_os = "win32")]
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


#[cfg(target_os = "win32")]
fn waitpid(pid: pid_t) -> c_int {
    ret rustrt::rust_process_wait(pid);
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "macos")]
fn waitpid(pid: pid_t) -> c_int {
    import libc::funcs::posix01::wait::*;
    let status = 0 as c_int;

    assert (waitpid(pid, ptr::mut_addr_of(status),
                    0 as c_int) != (-1 as c_int));
    ret status;
}


#[cfg(target_os = "linux")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "macos")]
fn pipe() -> {in: c_int, out: c_int} {
    let fds = {mut in: 0 as c_int,
               mut out: 0 as c_int };
    assert (libc::pipe(ptr::mut_addr_of(fds.in)) == (0 as c_int));
    ret {in: fds.in, out: fds.out};
}



#[cfg(target_os = "win32")]
fn pipe() -> {in: c_int, out: c_int} {
    // FIXME: remove this when export globs work properly.
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

    #[cfg(target_os = "macos")]
    #[cfg(target_os = "linux")]
    #[cfg(target_os = "freebsd")]
    fn pre() -> str { "lib" }

    #[cfg(target_os = "win32")]
    fn pre() -> str { "" }
}


fn self_exe_path() -> option<path> {

    #[cfg(target_os = "freebsd")]
    fn load_self() -> option<path> unsafe {
        import libc::funcs::bsd44::*;
        import libc::consts::os::extra::*;
        fill_charp_buf() {|buf, sz|
            let mib = [CTL_KERN as c_int,
                       KERN_PROC as c_int,
                       KERN_PROC_PATHNAME as c_int, -1 as c_int];
            sysctl(vec::unsafe::to_ptr(mib), vec::len(mib) as c_uint,
                   buf as *mut c_void, ptr::mut_addr_of(sz),
                   ptr::null(), 0u as size_t) == (0 as c_int)
        }
    }

    #[cfg(target_os = "linux")]
    fn load_self() -> option<path> unsafe {
        import libc::funcs::posix01::unistd::readlink;
        fill_charp_buf() {|buf, sz|
            as_c_charp("/proc/self/exe") { |proc_self_buf|
                readlink(proc_self_buf, buf, sz) != (-1 as ssize_t)
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn load_self() -> option<path> unsafe {
        // FIXME: remove imports when export globs work properly.
        import libc::funcs::extra::*;
        fill_charp_buf() {|buf, sz|
            _NSGetExecutablePath(buf, ptr::mut_addr_of(sz as u32))
                == (0 as c_int)
        }
    }

    #[cfg(target_os = "win32")]
    fn load_self() -> option<path> unsafe {
        // FIXME: remove imports when export globs work properly.
        import libc::types::os::arch::extra::*;
        import libc::funcs::extra::kernel32::*;
        import win32::*;
        fill_utf16_buf_and_decode() {|buf, sz|
            GetModuleFileNameW(0u, buf, sz)
        }
    }

    option::map(load_self()) {|pth|
        path::dirname(pth) + path::path_sep()
    }
}


#[doc = "
Returns the path to the user's home directory, if known.

On Unix, returns the value of the 'HOME' environment variable if it is set and
not equal to the empty string.

On Windows, returns the value of the 'HOME' environment variable if it is set
and not equal to the empty string. Otherwise, returns the value of the
'USERPROFILE' environment variable if it is set and not equal to the empty
string.

Otherwise, homedir returns option::none.
"]
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

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn secondary() -> option<path> {
        none
    }

    #[cfg(target_os = "win32")]
    fn secondary() -> option<path> {
        option::chain(getenv("USERPROFILE")) {|p|
            if !str::is_empty(p) {
                some(p)
            } else {
                none
            }
        }
    }
}



#[doc = "Indicates whether a path represents a directory"]
fn path_is_dir(p: path) -> bool {
    str::as_c_str(p) {|buf|
        rustrt::rust_path_is_dir(buf) != 0 as c_int
    }
}

#[doc = "Indicates whether a path exists"]
fn path_exists(p: path) -> bool {
    str::as_c_str(p) {|buf|
        rustrt::rust_path_exists(buf) != 0 as c_int
    }
}

// FIXME: under Windows, we should prepend the current drive letter to paths
// that start with a slash.
#[doc = "
Convert a relative path to an absolute path

If the given path is relative, return it prepended with the current working
directory. If the given path is already an absolute path, return it
as is.
"]
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


#[doc = "Creates a directory at the specified path"]
fn make_dir(p: path, mode: c_int) -> bool {
    ret mkdir(p, mode);

    #[cfg(target_os = "win32")]
    fn mkdir(p: path, _mode: c_int) -> bool unsafe {
        // FIXME: remove imports when export globs work properly.
        import libc::types::os::arch::extra::*;
        import libc::funcs::extra::kernel32::*;
        import win32::*;
        // FIXME: turn mode into something useful?
        as_utf16_p(p) {|buf|
            CreateDirectoryW(buf, unsafe::reinterpret_cast(0))
                != (0 as BOOL)
        }
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn mkdir(p: path, mode: c_int) -> bool {
        as_c_charp(p) {|c|
            libc::mkdir(c, mode as mode_t) == (0 as c_int)
        }
    }
}

#[doc = "Lists the contents of a directory"]
fn list_dir(p: path) -> [str] {

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn star(p: str) -> str { p }

    #[cfg(target_os = "win32")]
    fn star(p: str) -> str {
        let pl = str::len(p);
        if pl == 0u || (p[pl - 1u] as char != path::consts::path_sep
                        || p[pl - 1u] as char != path::consts::alt_path_sep) {
            p + path::path_sep() + "*"
        } else {
            p + "*"
        }
    }

    rustrt::rust_list_files(star(p)).filter {|filename|
        !str::eq(filename, ".") || !str::eq(filename, "..")
    }
}

#[doc = "
Lists the contents of a directory

This version prepends each entry with the directory.
"]
fn list_dir_path(p: path) -> [str] {
    let mut p = p;
    let pl = str::len(p);
    if pl == 0u || (p[pl - 1u] as char != path::consts::path_sep
                    && p[pl - 1u] as char != path::consts::alt_path_sep) {
        p += path::path_sep();
    }
    os::list_dir(p).map {|f| p + f}
}

#[doc = "Removes a directory at the specified path"]
fn remove_dir(p: path) -> bool {
   ret rmdir(p);

    #[cfg(target_os = "win32")]
    fn rmdir(p: path) -> bool {
        // FIXME: remove imports when export globs work properly.
        import libc::funcs::extra::kernel32::*;
        import libc::types::os::arch::extra::*;
        import win32::*;
        ret as_utf16_p(p) {|buf|
            RemoveDirectoryW(buf) != (0 as BOOL)
        };
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn rmdir(p: path) -> bool {
        ret as_c_charp(p) {|buf|
            libc::rmdir(buf) == (0 as c_int)
        };
    }
}

fn change_dir(p: path) -> bool {
    ret chdir(p);

    #[cfg(target_os = "win32")]
    fn chdir(p: path) -> bool {
        // FIXME: remove imports when export globs work properly.
        import libc::funcs::extra::kernel32::*;
        import libc::types::os::arch::extra::*;
        import win32::*;
        ret as_utf16_p(p) {|buf|
            SetCurrentDirectoryW(buf) != (0 as BOOL)
        };
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn chdir(p: path) -> bool {
        ret as_c_charp(p) {|buf|
            libc::chdir(buf) == (0 as c_int)
        };
    }
}

#[doc = "Deletes an existing file"]
fn remove_file(p: path) -> bool {
    ret unlink(p);

    #[cfg(target_os = "win32")]
    fn unlink(p: path) -> bool {
        // FIXME: remove imports when export globs work properly.
        import libc::funcs::extra::kernel32::*;
        import libc::types::os::arch::extra::*;
        import win32::*;
        ret as_utf16_p(p) {|buf|
            DeleteFileW(buf) != (0 as BOOL)
        };
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn unlink(p: path) -> bool {
        ret as_c_charp(p) {|buf|
            libc::unlink(buf) == (0 as c_int)
        };
    }
}



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




#[cfg(test)]
mod tests {


    fn make_rand_name() -> str {
        import rand;
        let rng: rand::rng = rand::rng();
        let n = "TEST" + rng.gen_str(10u);
        assert option::is_none(getenv(n));
        n
    }

    #[test]
    #[ignore(reason = "fails periodically on mac")]
    fn test_setenv() {
        let n = make_rand_name();
        setenv(n, "VALUE");
        assert getenv(n) == option::some("VALUE");
    }

    #[test]
    #[ignore(reason = "fails periodically on mac")]
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
    #[ignore(reason = "fails periodically on mac")]
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
    fn test_env_getenv() {
        let e = env();
        assert vec::len(e) > 0u;
        for vec::each(e) {|p|
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
    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn homedir() {
        let oldhome = getenv("HOME");

        setenv("HOME", "/home/MountainView");
        assert os::homedir() == some("/home/MountainView");

        setenv("HOME", "");
        assert os::homedir() == none;

        option::with_option_do(oldhome, {|s| setenv("HOME", s)});
    }

    #[test]
    #[cfg(target_os = "win32")]
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

        option::with_option_do(oldhome, {|s| setenv("HOME", s)});
        option::with_option_do(olduserprofile,
                               {|s| setenv("USERPROFILE", s)});
    }

    // Issue #712
    #[test]
    fn test_list_dir_no_invalid_memory_access() { os::list_dir("."); }

    #[test]
    fn list_dir() {
        let dirs = os::list_dir(".");
        // Just assuming that we've got some contents in the current directory
        assert (vec::len(dirs) > 0u);

        for vec::each(dirs) {|dir| log(debug, dir); }
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

}
