// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

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

use libc::{c_char, c_void, c_int, c_uint, size_t, ssize_t,
           mode_t, pid_t, FILE};
pub use libc::{close, fclose};

use option::{Some, None};

pub use consts::*;
use task::TaskBuilder;

// FIXME: move these to str perhaps? #2620

extern mod rustrt {
    fn rust_get_argc() -> c_int;
    fn rust_get_argv() -> **c_char;
    fn rust_getcwd() -> ~str;
    fn rust_path_is_dir(path: *libc::c_char) -> c_int;
    fn rust_path_exists(path: *libc::c_char) -> c_int;
    fn rust_list_files2(&&path: ~str) -> ~[~str];
    fn rust_process_wait(handle: c_int) -> c_int;
    fn last_os_error() -> ~str;
    fn rust_set_exit_status(code: libc::intptr_t);
}


const tmpbuf_sz : uint = 1000u;

pub fn getcwd() -> Path {
    Path(rustrt::rust_getcwd())
}

pub fn as_c_charp<T>(s: &str, f: fn(*c_char) -> T) -> T {
    str::as_c_str(s, |b| f(b as *c_char))
}

pub fn fill_charp_buf(f: fn(*mut c_char, size_t) -> bool)
    -> Option<~str> {
    let buf = vec::to_mut(vec::from_elem(tmpbuf_sz, 0u8 as c_char));
    do vec::as_mut_buf(buf) |b, sz| {
        if f(b, sz as size_t) unsafe {
            Some(str::raw::from_buf(b as *u8))
        } else {
            None
        }
    }
}

#[cfg(windows)]
mod win32 {
    use libc::DWORD;

    pub fn fill_utf16_buf_and_decode(f: fn(*mut u16, DWORD) -> DWORD)
        -> Option<~str> {
        let mut n = tmpbuf_sz as DWORD;
        let mut res = None;
        let mut done = false;
        while !done {
            let buf = vec::to_mut(vec::from_elem(n as uint, 0u16));
            do vec::as_mut_buf(buf) |b, _sz| {
                let k : DWORD = f(b, tmpbuf_sz as DWORD);
                if k == (0 as DWORD) {
                    done = true;
                } else if (k == n &&
                           libc::GetLastError() ==
                           libc::ERROR_INSUFFICIENT_BUFFER as DWORD) {
                    n *= (2 as DWORD);
                } else {
                    let sub = vec::slice(buf, 0u, k as uint);
                    res = option::Some(str::from_utf16(sub));
                    done = true;
                }
            }
        }
        return res;
    }

    pub fn as_utf16_p<T>(s: &str, f: fn(*u16) -> T) -> T {
        let mut t = str::to_utf16(s);
        // Null terminate before passing on.
        t += ~[0u16];
        vec::as_imm_buf(t, |buf, _len| f(buf))
    }
}

pub fn getenv(n: &str) -> Option<~str> {
    global_env::getenv(n)
}

pub fn setenv(n: &str, v: &str) {
    global_env::setenv(n, v)
}

pub fn env() -> ~[(~str,~str)] {
    global_env::env()
}

mod global_env {
    //! Internal module for serializing access to getenv/setenv

    extern mod rustrt {
        fn rust_global_env_chan_ptr() -> *libc::uintptr_t;
    }

    enum Msg {
        MsgGetEnv(~str, comm::Chan<Option<~str>>),
        MsgSetEnv(~str, ~str, comm::Chan<()>),
        MsgEnv(comm::Chan<~[(~str,~str)]>)
    }

    pub fn getenv(n: &str) -> Option<~str> {
        let env_ch = get_global_env_chan();
        let po = comm::Port();
        comm::send(env_ch, MsgGetEnv(str::from_slice(n),
                                     comm::Chan(&po)));
        comm::recv(po)
    }

    pub fn setenv(n: &str, v: &str) {
        let env_ch = get_global_env_chan();
        let po = comm::Port();
        comm::send(env_ch, MsgSetEnv(str::from_slice(n),
                                     str::from_slice(v),
                                     comm::Chan(&po)));
        comm::recv(po)
    }

    pub fn env() -> ~[(~str,~str)] {
        let env_ch = get_global_env_chan();
        let po = comm::Port();
        comm::send(env_ch, MsgEnv(comm::Chan(&po)));
        comm::recv(po)
    }

    fn get_global_env_chan() -> comm::Chan<Msg> {
        let global_ptr = rustrt::rust_global_env_chan_ptr();
        unsafe {
            private::chan_from_global_ptr(global_ptr, || {
                // FIXME (#2621): This would be a good place to use a very
                // small foreign stack
                task::task().sched_mode(task::SingleThreaded).unlinked()
            }, global_env_task)
        }
    }

    fn global_env_task(msg_po: comm::Port<Msg>) {
        unsafe {
            do private::weaken_task |weak_po| {
                loop {
                    match comm::select2(msg_po, weak_po) {
                      either::Left(MsgGetEnv(ref n, resp_ch)) => {
                        comm::send(resp_ch, impl_::getenv(*n))
                      }
                      either::Left(MsgSetEnv(ref n, ref v, resp_ch)) => {
                        comm::send(resp_ch, impl_::setenv(*n, *v))
                      }
                      either::Left(MsgEnv(resp_ch)) => {
                        comm::send(resp_ch, impl_::env())
                      }
                      either::Right(_) => break
                    }
                }
            }
        }
    }

    mod impl_ {
        extern mod rustrt {
            fn rust_env_pairs() -> ~[~str];
        }

        pub fn env() -> ~[(~str,~str)] {
            let mut pairs = ~[];
            for vec::each(rustrt::rust_env_pairs()) |p| {
                let vs = str::splitn_char(*p, '=', 1u);
                assert vec::len(vs) == 2u;
                pairs.push((copy vs[0], copy vs[1]));
            }
            move pairs
        }

        #[cfg(unix)]
        pub fn getenv(n: &str) -> Option<~str> {
            unsafe {
                let s = str::as_c_str(n, libc::getenv);
                return if ptr::null::<u8>() == cast::reinterpret_cast(&s) {
                    option::None::<~str>
                } else {
                    let s = cast::reinterpret_cast(&s);
                    option::Some::<~str>(str::raw::from_buf(s))
                };
            }
        }

        #[cfg(windows)]
        pub fn getenv(n: &str) -> Option<~str> {
            use win32::*;
            do as_utf16_p(n) |u| {
                do fill_utf16_buf_and_decode() |buf, sz| {
                    libc::GetEnvironmentVariableW(u, buf, sz)
                }
            }
        }


        #[cfg(unix)]
        pub fn setenv(n: &str, v: &str) {
            do str::as_c_str(n) |nbuf| {
                do str::as_c_str(v) |vbuf| {
                    libc::funcs::posix01::unistd::setenv(nbuf, vbuf, 1i32);
                }
            }
        }


        #[cfg(windows)]
        pub fn setenv(n: &str, v: &str) {
            use win32::*;
            do as_utf16_p(n) |nbuf| {
                do as_utf16_p(v) |vbuf| {
                    libc::SetEnvironmentVariableW(nbuf, vbuf);
                }
            }
        }

    }
}

pub fn fdopen(fd: c_int) -> *FILE {
    return do as_c_charp("r") |modebuf| {
        libc::fdopen(fd, modebuf)
    };
}


// fsync related

#[cfg(windows)]
pub fn fsync_fd(fd: c_int, _level: io::fsync::Level) -> c_int {
    use libc::funcs::extra::msvcrt::*;
    return commit(fd);
}

#[cfg(target_os = "linux")]
pub fn fsync_fd(fd: c_int, level: io::fsync::Level) -> c_int {
    use libc::funcs::posix01::unistd::*;
    match level {
      io::fsync::FSync
      | io::fsync::FullFSync => return fsync(fd),
      io::fsync::FDataSync => return fdatasync(fd)
    }
}

#[cfg(target_os = "macos")]
pub fn fsync_fd(fd: c_int, level: io::fsync::Level) -> c_int {
    use libc::consts::os::extra::*;
    use libc::funcs::posix88::fcntl::*;
    use libc::funcs::posix01::unistd::*;
    match level {
      io::fsync::FSync => return fsync(fd),
      _ => {
        // According to man fnctl, the ok retval is only specified to be !=-1
        if (fcntl(F_FULLFSYNC as c_int, fd) == -1 as c_int)
            { return -1 as c_int; }
        else
            { return 0 as c_int; }
      }
    }
}

#[cfg(target_os = "freebsd")]
pub fn fsync_fd(fd: c_int, _l: io::fsync::Level) -> c_int {
    use libc::funcs::posix01::unistd::*;
    return fsync(fd);
}


#[cfg(windows)]
pub fn waitpid(pid: pid_t) -> c_int {
    return rustrt::rust_process_wait(pid);
}

#[cfg(unix)]
pub fn waitpid(pid: pid_t) -> c_int {
    use libc::funcs::posix01::wait::*;
    let status = 0 as c_int;

    assert (waitpid(pid, ptr::mut_addr_of(&status),
                    0 as c_int) != (-1 as c_int));
    return status;
}


#[cfg(unix)]
pub fn pipe() -> {in: c_int, out: c_int} {
    let fds = {mut in: 0 as c_int,
               mut out: 0 as c_int };
    assert (libc::pipe(ptr::mut_addr_of(&(fds.in))) == (0 as c_int));
    return {in: fds.in, out: fds.out};
}



#[cfg(windows)]
pub fn pipe() -> {in: c_int, out: c_int} {
    // Windows pipes work subtly differently than unix pipes, and their
    // inheritance has to be handled in a different way that I do not fully
    // understand. Here we explicitly make the pipe non-inheritable, which
    // means to pass it to a subprocess they need to be duplicated first, as
    // in rust_run_program.
    let fds = { mut in: 0 as c_int,
                mut out: 0 as c_int };
    let res = libc::pipe(ptr::mut_addr_of(&(fds.in)),
                         1024 as c_uint,
                         (libc::O_BINARY | libc::O_NOINHERIT) as c_int);
    assert (res == 0 as c_int);
    assert (fds.in != -1 as c_int && fds.in != 0 as c_int);
    assert (fds.out != -1 as c_int && fds.in != 0 as c_int);
    return {in: fds.in, out: fds.out};
}

fn dup2(src: c_int, dst: c_int) -> c_int {
    libc::dup2(src, dst)
}


pub fn dll_filename(base: &str) -> ~str {
    return pre() + str::from_slice(base) + dll_suffix();

    #[cfg(unix)]
    fn pre() -> ~str { ~"lib" }

    #[cfg(windows)]
    fn pre() -> ~str { ~"" }
}


pub fn self_exe_path() -> Option<Path> {

    #[cfg(target_os = "freebsd")]
    fn load_self() -> Option<~str> {
        unsafe {
            use libc::funcs::bsd44::*;
            use libc::consts::os::extra::*;
            do fill_charp_buf() |buf, sz| {
                let mib = ~[CTL_KERN as c_int,
                           KERN_PROC as c_int,
                           KERN_PROC_PATHNAME as c_int, -1 as c_int];
                sysctl(vec::raw::to_ptr(mib), vec::len(mib) as c_uint,
                       buf as *mut c_void, ptr::mut_addr_of(&sz),
                       ptr::null(), 0u as size_t) == (0 as c_int)
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn load_self() -> Option<~str> {
        use libc::funcs::posix01::unistd::readlink;
        do fill_charp_buf() |buf, sz| {
            do as_c_charp("/proc/self/exe") |proc_self_buf| {
                readlink(proc_self_buf, buf, sz) != (-1 as ssize_t)
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn load_self() -> Option<~str> {
        do fill_charp_buf() |buf, sz| {
            libc::funcs::extra::_NSGetExecutablePath(
                buf, ptr::mut_addr_of(&(sz as u32))) == (0 as c_int)
        }
    }

    #[cfg(windows)]
    fn load_self() -> Option<~str> {
        use win32::*;
        do fill_utf16_buf_and_decode() |buf, sz| {
            libc::GetModuleFileNameW(0u as libc::DWORD, buf, sz)
        }
    }

    do load_self().map |pth| {
        Path(*pth).dir_path()
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
pub fn homedir() -> Option<Path> {
    return match getenv(~"HOME") {
        Some(ref p) => if !str::is_empty(*p) {
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
        do option::chain(getenv(~"USERPROFILE")) |p| {
            if !str::is_empty(p) {
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
            Some(move x) =>
                if str::is_empty(x) {
                    None
                } else {
                    Some(Path(x))
                },
            _ => None
        }
    }

    #[cfg(unix)]
    #[allow(non_implicitly_copyable_typarams)]
    fn lookup() -> Path {
        option::get_default(getenv_nonempty("TMPDIR"),
                            Path("/tmp"))
    }

    #[cfg(windows)]
    #[allow(non_implicitly_copyable_typarams)]
    fn lookup() -> Path {
        option::get_default(
                    option::or(getenv_nonempty("TMP"),
                    option::or(getenv_nonempty("TEMP"),
                    option::or(getenv_nonempty("USERPROFILE"),
                               getenv_nonempty("WINDIR")))),
                    Path("C:\\Windows"))
    }
}
/// Recursively walk a directory structure
pub fn walk_dir(p: &Path, f: fn(&Path) -> bool) {

    walk_dir_(p, f);

    fn walk_dir_(p: &Path, f: fn(&Path) -> bool) -> bool {
        let mut keepgoing = true;
        do list_dir(p).each |q| {
            let path = &p.push(*q);
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
        return keepgoing;
    }
}

/// Indicates whether a path represents a directory
pub fn path_is_dir(p: &Path) -> bool {
    do str::as_c_str(p.to_str()) |buf| {
        rustrt::rust_path_is_dir(buf) != 0 as c_int
    }
}

/// Indicates whether a path exists
pub fn path_exists(p: &Path) -> bool {
    do str::as_c_str(p.to_str()) |buf| {
        rustrt::rust_path_exists(buf) != 0 as c_int
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
        copy *p
    } else {
        getcwd().push_many(p.components)
    }
}


/// Creates a directory at the specified path
pub fn make_dir(p: &Path, mode: c_int) -> bool {
    return mkdir(p, mode);

    #[cfg(windows)]
    fn mkdir(p: &Path, _mode: c_int) -> bool {
        use win32::*;
        // FIXME: turn mode into something useful? #2623
        do as_utf16_p(p.to_str()) |buf| {
            libc::CreateDirectoryW(buf, unsafe {
                cast::reinterpret_cast(&0)
            })
                != (0 as libc::BOOL)
        }
    }

    #[cfg(unix)]
    fn mkdir(p: &Path, mode: c_int) -> bool {
        do as_c_charp(p.to_str()) |c| {
            libc::mkdir(c, mode as mode_t) == (0 as c_int)
        }
    }
}

/// Lists the contents of a directory
#[allow(non_implicitly_copyable_typarams)]
pub fn list_dir(p: &Path) -> ~[~str] {

    #[cfg(unix)]
    fn star(p: &Path) -> Path { copy *p }

    #[cfg(windows)]
    fn star(p: &Path) -> Path { p.push("*") }

    do rustrt::rust_list_files2(star(p).to_str()).filter |filename| {
        *filename != ~"." && *filename != ~".."
    }
}

/**
 * Lists the contents of a directory
 *
 * This version prepends each entry with the directory.
 */
pub fn list_dir_path(p: &Path) -> ~[~Path] {
    list_dir(p).map(|f| ~p.push(*f))
}

/// Removes a directory at the specified path
pub fn remove_dir(p: &Path) -> bool {
   return rmdir(p);

    #[cfg(windows)]
    fn rmdir(p: &Path) -> bool {
        use win32::*;
        return do as_utf16_p(p.to_str()) |buf| {
            libc::RemoveDirectoryW(buf) != (0 as libc::BOOL)
        };
    }

    #[cfg(unix)]
    fn rmdir(p: &Path) -> bool {
        return do as_c_charp(p.to_str()) |buf| {
            libc::rmdir(buf) == (0 as c_int)
        };
    }
}

pub fn change_dir(p: &Path) -> bool {
    return chdir(p);

    #[cfg(windows)]
    fn chdir(p: &Path) -> bool {
        use win32::*;
        return do as_utf16_p(p.to_str()) |buf| {
            libc::SetCurrentDirectoryW(buf) != (0 as libc::BOOL)
        };
    }

    #[cfg(unix)]
    fn chdir(p: &Path) -> bool {
        return do as_c_charp(p.to_str()) |buf| {
            libc::chdir(buf) == (0 as c_int)
        };
    }
}

/// Copies a file from one location to another
pub fn copy_file(from: &Path, to: &Path) -> bool {
    return do_copy_file(from, to);

    #[cfg(windows)]
    fn do_copy_file(from: &Path, to: &Path) -> bool {
        use win32::*;
        return do as_utf16_p(from.to_str()) |fromp| {
            do as_utf16_p(to.to_str()) |top| {
                libc::CopyFileW(fromp, top, (0 as libc::BOOL)) !=
                    (0 as libc::BOOL)
            }
        }
    }

    #[cfg(unix)]
    fn do_copy_file(from: &Path, to: &Path) -> bool {
        let istream = do as_c_charp(from.to_str()) |fromp| {
            do as_c_charp("rb") |modebuf| {
                libc::fopen(fromp, modebuf)
            }
        };
        if istream as uint == 0u {
            return false;
        }
        let ostream = do as_c_charp(to.to_str()) |top| {
            do as_c_charp("w+b") |modebuf| {
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
            do vec::as_mut_buf(buf) |b, _sz| {
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
        return ok;
    }
}

/// Deletes an existing file
pub fn remove_file(p: &Path) -> bool {
    return unlink(p);

    #[cfg(windows)]
    fn unlink(p: &Path) -> bool {
        use win32::*;
        return do as_utf16_p(p.to_str()) |buf| {
            libc::DeleteFileW(buf) != (0 as libc::BOOL)
        };
    }

    #[cfg(unix)]
    fn unlink(p: &Path) -> bool {
        return do as_c_charp(p.to_str()) |buf| {
            libc::unlink(buf) == (0 as c_int)
        };
    }
}

/// Get a string representing the platform-dependent last error
pub fn last_os_error() -> ~str {
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
pub fn set_exit_status(code: int) {
    rustrt::rust_set_exit_status(code as libc::intptr_t);
}

unsafe fn load_argc_and_argv(argc: c_int, argv: **c_char) -> ~[~str] {
    let mut args = ~[];
    for uint::range(0, argc as uint) |i| {
        vec::push(&mut args, str::raw::from_c_str(*argv.offset(i)));
    }
    move args
}

/**
 * Returns the command line arguments
 *
 * Returns a list of the command line arguments.
 */
#[cfg(target_os = "macos")]
fn real_args() -> ~[~str] {
    unsafe {
        let (argc, argv) = (*_NSGetArgc() as c_int,
                            *_NSGetArgv() as **c_char);
        load_argc_and_argv(argc, argv)
    }
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "freebsd")]
fn real_args() -> ~[~str] {
    unsafe {
        let argc = rustrt::rust_get_argc();
        let argv = rustrt::rust_get_argv();
        load_argc_and_argv(argc, argv)
    }
}

#[cfg(windows)]
fn real_args() -> ~[~str] {
    let mut nArgs: c_int = 0;
    let lpArgCount = ptr::to_mut_unsafe_ptr(&mut nArgs);
    let lpCmdLine = GetCommandLineW();
    let szArgList = CommandLineToArgvW(lpCmdLine, lpArgCount);

    let mut args = ~[];
    for uint::range(0, nArgs as uint) |i| {
        unsafe {
            // Determine the length of this argument.
            let ptr = *szArgList.offset(i);
            let mut len = 0;
            while *ptr.offset(len) != 0 { len += 1; }

            // Push it onto the list.
            vec::push(&mut args,
                      vec::raw::buf_as_slice(ptr, len,
                                             str::from_utf16));
        }
    }

    unsafe {
        LocalFree(cast::transmute(szArgList));
    }

    return args;
}

type LPCWSTR = *u16;

#[cfg(windows)]
#[link_name="kernel32"]
#[abi="stdcall"]
extern {
    fn GetCommandLineW() -> LPCWSTR;
    fn LocalFree(ptr: *c_void);
}

#[cfg(windows)]
#[link_name="shell32"]
#[abi="stdcall"]
extern {
    fn CommandLineToArgvW(lpCmdLine: LPCWSTR, pNumArgs: *mut c_int) -> **u16;
}

struct OverriddenArgs {
    val: ~[~str]
}

fn overridden_arg_key(_v: @OverriddenArgs) {}

pub fn args() -> ~[~str] {
    unsafe {
        match task::local_data::local_data_get(overridden_arg_key) {
            None => real_args(),
            Some(args) => copy args.val
        }
    }
}

pub fn set_args(new_args: ~[~str]) {
    unsafe {
        let overridden_args = @OverriddenArgs { val: copy new_args };
        task::local_data::local_data_set(overridden_arg_key, overridden_args);
    }
}

#[cfg(target_os = "macos")]
extern {
    // These functions are in crt_externs.h.
    pub fn _NSGetArgc() -> *c_int;
    pub fn _NSGetArgv() -> ***c_char;
}

#[cfg(unix)]
pub fn family() -> ~str { ~"unix" }

#[cfg(windows)]
pub fn family() -> ~str { ~"windows" }

#[cfg(target_os = "macos")]
mod consts {
    pub fn sysname() -> ~str { ~"macos" }
    pub fn exe_suffix() -> ~str { ~"" }
    pub fn dll_suffix() -> ~str { ~".dylib" }
}

#[cfg(target_os = "freebsd")]
mod consts {
    pub fn sysname() -> ~str { ~"freebsd" }
    pub fn exe_suffix() -> ~str { ~"" }
    pub fn dll_suffix() -> ~str { ~".so" }
}

#[cfg(target_os = "linux")]
mod consts {
    pub fn sysname() -> ~str { ~"linux" }
    pub fn exe_suffix() -> ~str { ~"" }
    pub fn dll_suffix() -> ~str { ~".so" }
}

#[cfg(target_os = "win32")]
mod consts {
    pub fn sysname() -> ~str { ~"win32" }
    pub fn exe_suffix() -> ~str { ~".exe" }
    pub fn dll_suffix() -> ~str { ~".dll" }
}

#[cfg(target_arch = "x86")]
pub fn arch() -> ~str { ~"x86" }

#[cfg(target_arch = "x86_64")]
pub fn arch() -> ~str { ~"x86_64" }

#[cfg(target_arch = "arm")]
pub fn arch() -> str { ~"arm" }

#[cfg(test)]
#[allow(non_implicitly_copyable_typarams)]
mod tests {

    #[test]
    pub fn last_os_error() {
        log(debug, last_os_error());
    }

    #[test]
    pub fn test_args() {
        let a = real_args();
        assert a.len() >= 1;
    }

    fn make_rand_name() -> ~str {
        let rng: rand::Rng = rand::Rng();
        let n = ~"TEST" + rng.gen_str(10u);
        assert getenv(n).is_none();
        move n
    }

    #[test]
    fn test_setenv() {
        let n = make_rand_name();
        setenv(n, ~"VALUE");
        assert getenv(n) == option::Some(~"VALUE");
    }

    #[test]
    #[ignore(cfg(windows))]
    #[ignore]
    fn test_setenv_overwrite() {
        let n = make_rand_name();
        setenv(n, ~"1");
        setenv(n, ~"2");
        assert getenv(n) == option::Some(~"2");
        setenv(n, ~"");
        assert getenv(n) == option::Some(~"");
    }

    // Windows GetEnvironmentVariable requires some extra work to make sure
    // the buffer the variable is copied into is the right size
    #[test]
    #[ignore(cfg(windows))]
    #[ignore]
    fn test_getenv_big() {
        let mut s = ~"";
        let mut i = 0;
        while i < 100 { s += ~"aaaaaaaaaa"; i += 1; }
        let n = make_rand_name();
        setenv(n, s);
        log(debug, s);
        assert getenv(n) == option::Some(move s);
    }

    #[test]
    fn test_self_exe_path() {
        let path = os::self_exe_path();
        assert path.is_some();
        let path = path.get();
        log(debug, path);

        // Hard to test this function
        assert path.is_absolute;
    }

    #[test]
    #[ignore]
    fn test_env_getenv() {
        let e = env();
        assert vec::len(e) > 0u;
        for vec::each(e) |p| {
            let (n, v) = copy *p;
            log(debug, n);
            let v2 = getenv(n);
            // MingW seems to set some funky environment variables like
            // "=C:=C:\MinGW\msys\1.0\bin" and "!::=::\" that are returned
            // from env() but not visible from getenv().
            assert v2.is_none() || v2 == option::Some(move v);
        }
    }

    #[test]
    fn test_env_setenv() {
        let n = make_rand_name();

        let mut e = env();
        setenv(n, ~"VALUE");
        assert !vec::contains(e, &(copy n, ~"VALUE"));

        e = env();
        assert vec::contains(e, &(move n, ~"VALUE"));
    }

    #[test]
    fn test() {
        assert (!Path("test-path").is_absolute);

        log(debug, ~"Current working directory: " + getcwd().to_str());

        log(debug, make_absolute(&Path("test-path")));
        log(debug, make_absolute(&Path("/usr/bin")));
    }

    #[test]
    #[cfg(unix)]
    fn homedir() {
        let oldhome = getenv(~"HOME");

        setenv(~"HOME", ~"/home/MountainView");
        assert os::homedir() == Some(Path("/home/MountainView"));

        setenv(~"HOME", ~"");
        assert os::homedir().is_none();

        oldhome.iter(|s| setenv(~"HOME", *s));
    }

    #[test]
    #[cfg(windows)]
    fn homedir() {

        let oldhome = getenv(~"HOME");
        let olduserprofile = getenv(~"USERPROFILE");

        setenv(~"HOME", ~"");
        setenv(~"USERPROFILE", ~"");

        assert os::homedir().is_none();

        setenv(~"HOME", ~"/home/MountainView");
        assert os::homedir() == Some(Path("/home/MountainView"));

        setenv(~"HOME", ~"");

        setenv(~"USERPROFILE", ~"/home/MountainView");
        assert os::homedir() == Some(Path("/home/MountainView"));

        setenv(~"HOME", ~"/home/MountainView");
        setenv(~"USERPROFILE", ~"/home/PaloAlto");
        assert os::homedir() == Some(Path("/home/MountainView"));

        option::iter(&oldhome, |s| setenv(~"HOME", *s));
        option::iter(&olduserprofile,
                               |s| setenv(~"USERPROFILE", *s));
    }

    #[test]
    fn tmpdir() {
        assert !str::is_empty(os::tmpdir().to_str());
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
        assert (vec::len(dirs) > 0u);

        for vec::each(dirs) |dir| {
            log(debug, *dir);
        }
    }

    #[test]
    fn path_is_dir() {
        assert (os::path_is_dir(&Path(".")));
        assert (!os::path_is_dir(&Path("test/stdtest/fs.rs")));
    }

    #[test]
    fn path_exists() {
        assert (os::path_exists(&Path(".")));
        assert (!os::path_exists(&Path("test/nonexistent-bogus-path")));
    }

    #[test]
    fn copy_file_does_not_exist() {
      assert !os::copy_file(&Path("test/nonexistent-bogus-path"),
                            &Path("test/other-bogus-path"));
      assert !os::path_exists(&Path("test/other-bogus-path"));
    }

    #[test]
    fn copy_file_ok() {
      let tempdir = getcwd(); // would like to use $TMPDIR,
                              // doesn't seem to work on Linux
      assert (str::len(tempdir.to_str()) > 0u);
      let in = tempdir.push("in.txt");
      let out = tempdir.push("out.txt");

      /* Write the temp input file */
        let ostream = do as_c_charp(in.to_str()) |fromp| {
            do as_c_charp("w+b") |modebuf| {
                libc::fopen(fromp, modebuf)
            }
      };
      assert (ostream as uint != 0u);
      let s = ~"hello";
      let mut buf = vec::to_mut(str::to_bytes(s) + ~[0 as u8]);
      do vec::as_mut_buf(buf) |b, _len| {
          assert (libc::fwrite(b as *c_void, 1u as size_t,
                               (str::len(s) + 1u) as size_t, ostream)
                  == buf.len() as size_t)};
      assert (libc::fclose(ostream) == (0u as c_int));
      let rs = os::copy_file(&in, &out);
      if (!os::path_exists(&in)) {
        fail (fmt!("%s doesn't exist", in.to_str()));
      }
      assert(rs);
      let rslt = run::run_program(~"diff", ~[in.to_str(), out.to_str()]);
      assert (rslt == 0);
      assert (remove_file(&in));
      assert (remove_file(&out));
    }
}
