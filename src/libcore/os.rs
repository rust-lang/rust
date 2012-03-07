// Higher-level interfaces to libc::* functions and operating system services.
//
// In general these take and return rust types, use rust idioms (enums,
// closures, vectors) rather than C idioms, and do more extensive safety
// checks.
//
// This module is not meant to only contain 1:1 mappings to libc entries; any
// os-interface code that is reasonably useful and broadly applicable can go
// here. Including utility routines that merely build on other os code.
//
// We assume the general case is that users do not care, and do not want to
// be made to care, which operating system they are on. While they may want
// to special case various special cases -- and so we will not _hide_ the
// facts of which OS the user is on -- they should be given the opportunity
// to write OS-ignorant code by default.

import libc::{c_char, c_void, c_int, c_uint, size_t, mode_t, pid_t, FILE};
import libc::{close, fclose};

import getcwd = rustrt::rust_getcwd;
import consts::*;

export close, fclose;
export env, getenv, setenv, fdopen, pipe;
export getcwd, dll_filename, self_exe_path;
export exe_suffix, dll_suffix, sysname;
export homedir, list_dir, path_is_dir, path_exists;

native mod rustrt {
    fn rust_env_pairs() -> [str];
    fn rust_getcwd() -> str;
    fn rust_path_is_dir(path: str::sbuf) -> c_int;
    fn rust_path_exists(path: str::sbuf) -> c_int;
    fn rust_list_files(path: str) -> [str];
    fn rust_process_wait(handle: c_int) -> c_int;
}


fn env() -> [(str,str)] {
    let pairs = [];
    for p in rustrt::rust_env_pairs() {
        let vs = str::splitn_char(p, '=', 1u);
        assert vec::len(vs) == 2u;
        pairs += [(vs[0], vs[1])];
    }
    ret pairs;
}

fn as_c_charp<T>(s: str, f: fn(*c_char) -> T) -> T {
    str::as_buf(s) {|b| f(b as *c_char) }
}

fn as_utf16_p<T>(s: str, f: fn(*u16) -> T) -> T {
    let t = str::to_utf16(s);
    // "null terminate"
    t += [0u16];
    vec::as_buf(t, f)
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
            option::some::<str>(str::from_cstr(s))
        };
}

#[cfg(target_os = "win32")]
fn getenv(n: str) -> option<str> unsafe {
    import libc::types::os::arch::extra::*;
    import libc::funcs::extra::kernel32;
    as_utf16_p(n) {|u|
        let bufsize = 1023u;
        let buf = vec::to_mut(vec::init_elt(bufsize, 0u16));
        vec::as_mut_buf(buf) {|b|
            let k = kernel32::GetEnvironmentVariableW(u, b,
                                                      bufsize as DWORD);
            if k != (0 as DWORD) {
                let sub = vec::slice(buf, 0u, k as uint);
                option::some::<str>(str::from_utf16(sub))
            } else {
                option::none::<str>
            }
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
    import libc::funcs::extra::kernel32;
    as_utf16_p(n) {|nbuf|
        as_utf16_p(v) {|vbuf|
            kernel32::SetEnvironmentVariableW(nbuf, vbuf);
        }
    }
}


fn fdopen(fd: c_int) -> *FILE {
    ret as_c_charp("r") {|modebuf|
        libc::fdopen(fd, modebuf)
    };
}


// fsync related

enum fsync_level {
    // whatever fsync does on that platform
    fsync,

    // fdatasync on linux, similiar or more on other platforms
    fdatasync,

    // full fsync
    //
    // You must additionally sync the parent directory as well!
    fullfsync,
}

#[cfg(target_os = "win32")]
fn fsync_fd(fd: c_int, _level: fsync_level) -> c_int {
    import libc::funcs::extra::msvcrt::*;
    ret commit(fd);
}

#[cfg(target_os = "linux")]
fn fsync_fd(fd: c_int, level: fsync_level) -> c_int {
    import libc::funcs::posix01::unistd::*;
    alt level {
      fsync | fullfsync { ret fsync(fd); }
      fdatasync { ret fdatasync(fd); }
    }
}

#[cfg(target_os = "macos")]
fn fsync_fd(fd: c_int, level: fsync_level) -> c_int {
    import libc::consts::os::extra::*;
    import libc::funcs::posix88::fcntl::*;
    import libc::funcs::posix01::unistd::*;
    alt level {
      fsync { ret fsync(fd); }
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
fn fsync_fd(fd: c_int, _l: fsync_level) -> c_int {
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
    let fds = {mutable in: 0 as c_int,
               mutable out: 0 as c_int };
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
    let fds = { mutable in: 0 as c_int,
               mutable out: 0 as c_int };
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

fn self_exe_path() -> option<path> unsafe {
    let bufsize = 1023u;
    let buf = vec::to_mut(vec::init_elt(bufsize, 0u8 as c_char));
    // FIXME: This does not handle the case where the buffer is too small
    ret vec::as_mut_buf(buf) {|pbuf|
        if load_self(pbuf as *mutable c_char, bufsize as c_uint) {
            let path = str::from_cstr(pbuf as str::sbuf);
            option::some(path::dirname(path) + path::path_sep())
        } else {
            option::none
        }
    };

    #[cfg(target_os = "freebsd")]
    unsafe fn load_self(pth: *mutable c_char, plen: c_uint) -> bool {
        // FIXME: remove imports when export globs work properly.
        import libc::funcs::bsd44::*;
        import libc::consts::os::extra::*;
        let mib = [CTL_KERN as c_int,
                   KERN_PROC as c_int,
                   KERN_PROC_PATHNAME as c_int, -1 as c_int];
        ret sysctl(vec::unsafe::to_ptr(mib), vec::len(mib) as c_uint,
                   pth as *mutable c_void, ptr::mut_addr_of(plen as size_t),
                   ptr::null(), 0u as size_t)
            == (0 as c_int);
    }

    #[cfg(target_os = "linux")]
    unsafe fn load_self(pth: *mutable c_char, plen: c_uint) -> bool {
        // FIXME: remove imports when export globs work properly.
        import libc::funcs::posix01::unistd::readlink;
        as_c_charp("/proc/self/exe") { |proc_self_buf|
            ret readlink(proc_self_buf, pth, plen as size_t) != -1;
        }
    }

    #[cfg(target_os = "win32")]
    unsafe fn load_self(pth: *mutable c_char, plen: c_uint) -> bool {
        // FIXME: remove imports when export globs work properly.
        import libc::types::os::arch::extra::*;
        import libc::funcs::extra::kernel32;
        ret kernel32::GetModuleFileNameA(0u, pth, plen) != (0 as DWORD);
    }

    #[cfg(target_os = "macos")]
    unsafe fn load_self(pth: *mutable c_char, plen: c_uint) -> bool {
        // FIXME: remove imports when export globs work properly.
        import libc::funcs::extra::*;
        let mplen = plen;
        ret _NSGetExecutablePath(pth, ptr::mut_addr_of(mplen))
            == (0 as c_int);
    }

}


/*
Function: homedir

Returns the path to the user's home directory, if known.

On Unix, returns the value of the "HOME" environment variable if it is set and
not equal to the empty string.

On Windows, returns the value of the "HOME" environment variable if it is set
and not equal to the empty string. Otherwise, returns the value of the
"USERPROFILE" environment variable if it is set and not equal to the empty
string.

Otherwise, homedir returns option::none.
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

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn secondary() -> option<path> {
        none
    }

    #[cfg(target_os = "win32")]
    fn secondary() -> option<path> {
        option::maybe(none, getenv("USERPROFILE")) {|p|
            if !str::is_empty(p) {
                some(p)
            } else {
                none
            }
        }
    }
}



/*
Function: path_is_dir

Indicates whether a path represents a directory.
*/
fn path_is_dir(p: path) -> bool {
    ret str::as_buf(p, {|buf|
        rustrt::rust_path_is_dir(buf) != 0 as c_int
    });
}

/*
Function: path_exists

Indicates whether a path exists.
*/
fn path_exists(p: path) -> bool {
    ret str::as_buf(p, {|buf|
        rustrt::rust_path_exists(buf) != 0 as c_int
    });
}

// FIXME: under Windows, we should prepend the current drive letter to paths
// that start with a slash.
/*
Function: make_absolute

Convert a relative path to an absolute path

If the given path is relative, return it prepended with the current working
directory. If the given path is already an absolute path, return it
as is.
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


/*
Function: make_dir

Creates a directory at the specified path.
*/
fn make_dir(p: path, mode: c_int) -> bool {
    ret mkdir(p, mode);

    #[cfg(target_os = "win32")]
    fn mkdir(_p: path, _mode: c_int) -> bool unsafe {
        // FIXME: turn mode into something useful?
        ret as_c_charp(_p, {|buf|
            // FIXME: remove imports when export globs work properly.
            import libc::types::os::arch::extra::*;
            import libc::funcs::extra::kernel32;
            kernel32::CreateDirectoryA(
                buf, unsafe::reinterpret_cast(0)) != (0 as BOOL)
        });
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn mkdir(p: path, mode: c_int) -> bool {
        ret as_c_charp(p) {|c|
            libc::mkdir(c, mode as mode_t) == (0 as c_int)
        };
    }
}

/*
Function: list_dir

Lists the contents of a directory.
*/
fn list_dir(p: path) -> [str] {

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn star() -> str { "" }

    #[cfg(target_os = "win32")]
    fn star() -> str { "*" }

    let p = p;
    let pl = str::len(p);
    if pl == 0u || (p[pl - 1u] as char != path::consts::path_sep
                    && p[pl - 1u] as char != path::consts::alt_path_sep) {
        p += path::path_sep();
    }
    let full_paths: [str] = [];
    for filename: str in rustrt::rust_list_files(p + star()) {
        if !str::eq(filename, ".") {
            if !str::eq(filename, "..") {
                full_paths += [p + filename];
            }
        }
    }
    ret full_paths;
}

/*
Function: remove_dir

Removes a directory at the specified path.
*/
fn remove_dir(p: path) -> bool {
   ret rmdir(p);

    #[cfg(target_os = "win32")]
    fn rmdir(p: path) -> bool {
        // FIXME: remove imports when export globs work properly.
        import libc::funcs::extra::kernel32;
        import libc::types::os::arch::extra::*;
        ret as_c_charp(p) {|buf|
            kernel32::RemoveDirectoryA(buf) != (0 as BOOL)
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
        import libc::funcs::extra::kernel32;
        import libc::types::os::arch::extra::*;
        ret as_c_charp(p) {|buf|
            kernel32::SetCurrentDirectoryA(buf) != (0 as BOOL)
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

/*
Function: remove_file

Deletes an existing file.
*/
fn remove_file(p: path) -> bool {
    ret unlink(p);

    #[cfg(target_os = "win32")]
    fn unlink(p: path) -> bool {
        // FIXME: remove imports when export globs work properly.
        import libc::funcs::extra::kernel32;
        import libc::types::os::arch::extra::*;
        ret as_c_charp(p) {|buf|
            kernel32::DeleteFileA(buf) != (0 as BOOL)
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

        option::may(oldhome, {|s| setenv("HOME", s)});
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

        option::may(oldhome, {|s| setenv("HOME", s)});
        option::may(olduserprofile, {|s| setenv("USERPROFILE", s)});
    }

    // Issue #712
    #[test]
    fn test_list_dir_no_invalid_memory_access() { os::list_dir("."); }

    #[test]
    fn list_dir() {
        let dirs = os::list_dir(".");
        // Just assuming that we've got some contents in the current directory
        assert (vec::len(dirs) > 0u);

        for dir in dirs { log(debug, dir); }
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