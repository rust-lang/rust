// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;

use collections::HashMap;
use env;
use ffi::{OsString, OsStr, CString};
use fmt;
use hash::Hash;
use io::{self, Error, ErrorKind};
use libc::{self, pid_t, c_void, c_int, gid_t, uid_t};
use mem;
use old_io;
use os;
use os::unix::OsStrExt;
use ptr;
use sync::mpsc::{channel, Sender, Receiver};
use sys::pipe2::AnonPipe;
use sys::{self, retry, c, wouldblock, set_nonblocking, ms_to_timeval, cvt};
use sys_common::AsInner;

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct Command {
    pub program: CString,
    pub args: Vec<CString>,
    pub env: Option<HashMap<OsString, OsString>>,
    pub cwd: Option<CString>,
    pub uid: Option<uid_t>,
    pub gid: Option<gid_t>,
    pub detach: bool, // not currently exposed in std::process
}

impl Command {
    pub fn new(program: &OsStr) -> Command {
        Command {
            program: program.to_cstring().unwrap(),
            args: Vec::new(),
            env: None,
            cwd: None,
            uid: None,
            gid: None,
            detach: false,
        }
    }

    pub fn arg(&mut self, arg: &OsStr) {
        self.args.push(arg.to_cstring().unwrap())
    }
    pub fn args<'a, I: Iterator<Item = &'a OsStr>>(&mut self, args: I) {
        self.args.extend(args.map(|s| OsStrExt::to_cstring(s).unwrap()))
    }
    fn init_env_map(&mut self) {
        if self.env.is_none() {
            self.env = Some(env::vars_os().collect());
        }
    }
    pub fn env(&mut self, key: &OsStr, val: &OsStr) {
        self.init_env_map();
        self.env.as_mut().unwrap().insert(key.to_os_string(), val.to_os_string());
    }
    pub fn env_remove(&mut self, key: &OsStr) {
        self.init_env_map();
        self.env.as_mut().unwrap().remove(&key.to_os_string());
    }
    pub fn env_clear(&mut self) {
        self.env = Some(HashMap::new())
    }
    pub fn cwd(&mut self, dir: &OsStr) {
        self.cwd = Some(dir.to_cstring().unwrap())
    }
}

////////////////////////////////////////////////////////////////////////////////
// Processes
////////////////////////////////////////////////////////////////////////////////

/// Unix exit statuses
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum ExitStatus {
    /// Normal termination with an exit code.
    Code(i32),

    /// Termination by signal, with the signal number.
    ///
    /// Never generated on Windows.
    Signal(i32),
}

impl ExitStatus {
    pub fn success(&self) -> bool {
        *self == ExitStatus::Code(0)
    }
    pub fn code(&self) -> Option<i32> {
        match *self {
            ExitStatus::Code(c) => Some(c),
            _ => None
        }
    }
}

impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ExitStatus::Code(code) =>  write!(f, "exit code: {}", code),
            ExitStatus::Signal(code) =>  write!(f, "signal: {}", code),
        }
    }
}

/// The unique id of the process (this should never be negative).
pub struct Process {
    pid: pid_t
}

const CLOEXEC_MSG_FOOTER: &'static [u8] = b"NOEX";

impl Process {
    pub fn id(&self) -> pid_t {
        self.pid
    }

    pub unsafe fn kill(&self) -> io::Result<()> {
        try!(cvt(libc::funcs::posix88::signal::kill(self.pid, libc::SIGKILL)));
        Ok(())
    }

    pub fn spawn(cfg: &Command,
                 in_fd: Option<AnonPipe>, out_fd: Option<AnonPipe>, err_fd: Option<AnonPipe>)
                 -> io::Result<Process>
    {
        use libc::funcs::posix88::unistd::{fork, dup2, close, chdir, execvp};
        use libc::funcs::bsd44::getdtablesize;

        mod rustrt {
            extern {
                pub fn rust_unset_sigprocmask();
            }
        }

        unsafe fn set_cloexec(fd: c_int) {
            let ret = c::ioctl(fd, c::FIOCLEX);
            assert_eq!(ret, 0);
        }

        let dirp = cfg.cwd.as_ref().map(|c| c.as_ptr()).unwrap_or(ptr::null());

        with_envp(cfg.env.as_ref(), |envp: *const c_void| {
            with_argv(&cfg.program, &cfg.args, |argv: *const *const libc::c_char| unsafe {
                let (input, mut output) = try!(sys::pipe2::anon_pipe());

                // We may use this in the child, so perform allocations before the
                // fork
                let devnull = b"/dev/null\0";

                set_cloexec(output.raw());

                let pid = fork();
                if pid < 0 {
                    return Err(Error::last_os_error())
                } else if pid > 0 {
                    #[inline]
                    fn combine(arr: &[u8]) -> i32 {
                        let a = arr[0] as u32;
                        let b = arr[1] as u32;
                        let c = arr[2] as u32;
                        let d = arr[3] as u32;

                        ((a << 24) | (b << 16) | (c << 8) | (d << 0)) as i32
                    }

                    let p = Process{ pid: pid };
                    drop(output);
                    let mut bytes = [0; 8];

                    // loop to handle EINTER
                    loop {
                        match input.read(&mut bytes) {
                            Ok(8) => {
                                assert!(combine(CLOEXEC_MSG_FOOTER) == combine(&bytes[4.. 8]),
                                        "Validation on the CLOEXEC pipe failed: {:?}", bytes);
                                let errno = combine(&bytes[0.. 4]);
                                assert!(p.wait().is_ok(),
                                        "wait() should either return Ok or panic");
                                return Err(Error::from_os_error(errno))
                            }
                            Ok(0) => return Ok(p),
                            Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
                            Err(e) => {
                                assert!(p.wait().is_ok(),
                                        "wait() should either return Ok or panic");
                                panic!("the CLOEXEC pipe failed: {:?}", e)
                            },
                            Ok(..) => { // pipe I/O up to PIPE_BUF bytes should be atomic
                                assert!(p.wait().is_ok(),
                                        "wait() should either return Ok or panic");
                                panic!("short read on the CLOEXEC pipe")
                            }
                        }
                    }
                }

                // And at this point we've reached a special time in the life of the
                // child. The child must now be considered hamstrung and unable to
                // do anything other than syscalls really. Consider the following
                // scenario:
                //
                //      1. Thread A of process 1 grabs the malloc() mutex
                //      2. Thread B of process 1 forks(), creating thread C
                //      3. Thread C of process 2 then attempts to malloc()
                //      4. The memory of process 2 is the same as the memory of
                //         process 1, so the mutex is locked.
                //
                // This situation looks a lot like deadlock, right? It turns out
                // that this is what pthread_atfork() takes care of, which is
                // presumably implemented across platforms. The first thing that
                // threads to *before* forking is to do things like grab the malloc
                // mutex, and then after the fork they unlock it.
                //
                // Despite this information, libnative's spawn has been witnessed to
                // deadlock on both OSX and FreeBSD. I'm not entirely sure why, but
                // all collected backtraces point at malloc/free traffic in the
                // child spawned process.
                //
                // For this reason, the block of code below should contain 0
                // invocations of either malloc of free (or their related friends).
                //
                // As an example of not having malloc/free traffic, we don't close
                // this file descriptor by dropping the FileDesc (which contains an
                // allocation). Instead we just close it manually. This will never
                // have the drop glue anyway because this code never returns (the
                // child will either exec() or invoke libc::exit)
                let _ = libc::close(input.raw());

                fn fail(output: &mut AnonPipe) -> ! {
                    let errno = sys::os::errno() as u32;
                    let bytes = [
                        (errno >> 24) as u8,
                        (errno >> 16) as u8,
                        (errno >>  8) as u8,
                        (errno >>  0) as u8,
                        CLOEXEC_MSG_FOOTER[0], CLOEXEC_MSG_FOOTER[1],
                        CLOEXEC_MSG_FOOTER[2], CLOEXEC_MSG_FOOTER[3]
                    ];
                    // pipe I/O up to PIPE_BUF bytes should be atomic
                    assert!(output.write(&bytes).is_ok());
                    unsafe { libc::_exit(1) }
                }

                rustrt::rust_unset_sigprocmask();

                // If a stdio file descriptor is set to be ignored, we don't
                // actually close it, but rather open up /dev/null into that
                // file descriptor. Otherwise, the first file descriptor opened
                // up in the child would be numbered as one of the stdio file
                // descriptors, which is likely to wreak havoc.
                let setup = |&: src: Option<AnonPipe>, dst: c_int| {
                    let src = match src {
                        None => {
                            let flags = if dst == libc::STDIN_FILENO {
                                libc::O_RDONLY
                            } else {
                                libc::O_RDWR
                            };
                            libc::open(devnull.as_ptr() as *const _, flags, 0)
                        }
                        Some(obj) => {
                            let fd = obj.raw();
                            // Leak the memory and the file descriptor. We're in the
                            // child now an all our resources are going to be
                            // cleaned up very soon
                            mem::forget(obj);
                            fd
                        }
                    };
                    src != -1 && retry(|| dup2(src, dst)) != -1
                };

                if !setup(in_fd, libc::STDIN_FILENO) { fail(&mut output) }
                if !setup(out_fd, libc::STDOUT_FILENO) { fail(&mut output) }
                if !setup(err_fd, libc::STDERR_FILENO) { fail(&mut output) }

                // close all other fds
                for fd in (3..getdtablesize()).rev() {
                    if fd != output.raw() {
                        let _ = close(fd as c_int);
                    }
                }

                match cfg.gid {
                    Some(u) => {
                        if libc::setgid(u as libc::gid_t) != 0 {
                            fail(&mut output);
                        }
                    }
                    None => {}
                }
                match cfg.uid {
                    Some(u) => {
                        // When dropping privileges from root, the `setgroups` call
                        // will remove any extraneous groups. If we don't call this,
                        // then even though our uid has dropped, we may still have
                        // groups that enable us to do super-user things. This will
                        // fail if we aren't root, so don't bother checking the
                        // return value, this is just done as an optimistic
                        // privilege dropping function.
                        extern {
                            fn setgroups(ngroups: libc::c_int,
                                         ptr: *const libc::c_void) -> libc::c_int;
                        }
                        let _ = setgroups(0, ptr::null());

                        if libc::setuid(u as libc::uid_t) != 0 {
                            fail(&mut output);
                        }
                    }
                    None => {}
                }
                if cfg.detach {
                    // Don't check the error of setsid because it fails if we're the
                    // process leader already. We just forked so it shouldn't return
                    // error, but ignore it anyway.
                    let _ = libc::setsid();
                }
                if !dirp.is_null() && chdir(dirp) == -1 {
                    fail(&mut output);
                }
                if !envp.is_null() {
                    *sys::os::environ() = envp as *const _;
                }
                let _ = execvp(*argv, argv as *mut _);
                fail(&mut output);
            })
        })
    }

    pub fn wait(&self) -> io::Result<ExitStatus> {
        let mut status = 0 as c_int;
        try!(cvt(retry(|| unsafe { c::waitpid(self.pid, &mut status, 0) })));
        Ok(translate_status(status))
    }

    pub fn try_wait(&self) -> Option<ExitStatus> {
        let mut status = 0 as c_int;
        match retry(|| unsafe {
            c::waitpid(self.pid, &mut status, c::WNOHANG)
        }) {
            n if n == self.pid => Some(translate_status(status)),
            0 => None,
            n => panic!("unknown waitpid error `{:?}`: {:?}", n,
                       super::last_error()),
        }
    }
}

fn with_argv<T,F>(prog: &CString, args: &[CString], cb: F) -> T
    where F : FnOnce(*const *const libc::c_char) -> T
{
    let mut ptrs: Vec<*const libc::c_char> = Vec::with_capacity(args.len()+1);

    // Convert the CStrings into an array of pointers. Note: the
    // lifetime of the various CStrings involved is guaranteed to be
    // larger than the lifetime of our invocation of cb, but this is
    // technically unsafe as the callback could leak these pointers
    // out of our scope.
    ptrs.push(prog.as_ptr());
    ptrs.extend(args.iter().map(|tmp| tmp.as_ptr()));

    // Add a terminating null pointer (required by libc).
    ptrs.push(ptr::null());

    cb(ptrs.as_ptr())
}

fn with_envp<T, F>(env: Option<&HashMap<OsString, OsString>>, cb: F) -> T
    where F : FnOnce(*const c_void) -> T
{
    // On posixy systems we can pass a char** for envp, which is a
    // null-terminated array of "k=v\0" strings. Since we must create
    // these strings locally, yet expose a raw pointer to them, we
    // create a temporary vector to own the CStrings that outlives the
    // call to cb.
    match env {
        Some(env) => {
            let mut tmps = Vec::with_capacity(env.len());

            for pair in env {
                let mut kv = Vec::new();
                kv.push_all(pair.0.as_bytes());
                kv.push('=' as u8);
                kv.push_all(pair.1.as_bytes());
                kv.push(0); // terminating null
                tmps.push(kv);
            }

            // As with `with_argv`, this is unsafe, since cb could leak the pointers.
            let mut ptrs: Vec<*const libc::c_char> =
                tmps.iter()
                    .map(|tmp| tmp.as_ptr() as *const libc::c_char)
                    .collect();
            ptrs.push(ptr::null());

            cb(ptrs.as_ptr() as *const c_void)
        }
        _ => cb(ptr::null())
    }
}

fn translate_status(status: c_int) -> ExitStatus {
    #![allow(non_snake_case)]
    #[cfg(any(target_os = "linux", target_os = "android"))]
    mod imp {
        pub fn WIFEXITED(status: i32) -> bool { (status & 0xff) == 0 }
        pub fn WEXITSTATUS(status: i32) -> i32 { (status >> 8) & 0xff }
        pub fn WTERMSIG(status: i32) -> i32 { status & 0x7f }
    }

    #[cfg(any(target_os = "macos",
              target_os = "ios",
              target_os = "freebsd",
              target_os = "dragonfly",
              target_os = "openbsd"))]
    mod imp {
        pub fn WIFEXITED(status: i32) -> bool { (status & 0x7f) == 0 }
        pub fn WEXITSTATUS(status: i32) -> i32 { status >> 8 }
        pub fn WTERMSIG(status: i32) -> i32 { status & 0o177 }
    }

    if imp::WIFEXITED(status) {
        ExitStatus::Code(imp::WEXITSTATUS(status))
    } else {
        ExitStatus::Signal(imp::WTERMSIG(status))
    }
}
