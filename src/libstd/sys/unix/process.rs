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
use self::Req::*;

use collections::HashMap;
#[cfg(stage0)]
use collections::hash_map::Hasher;
use ffi::CString;
use hash::Hash;
use old_io::process::{ProcessExit, ExitStatus, ExitSignal};
use old_io::{self, IoResult, IoError, EndOfFile};
use libc::{self, pid_t, c_void, c_int};
use mem;
use os;
use old_path::BytesContainer;
use ptr;
use sync::mpsc::{channel, Sender, Receiver};
use sys::fs::FileDesc;
use sys::{self, retry, c, wouldblock, set_nonblocking, ms_to_timeval};
use sys_common::helper_thread::Helper;
use sys_common::{AsInner, mkerr_libc, timeout};

pub use sys_common::ProcessConfig;

helper_init! { static HELPER: Helper<Req> }

/// Unix-specific extensions to the Command builder
pub struct CommandExt {
    uid: Option<u32>,
    gid: Option<u32>,
}

/// The unique id of the process (this should never be negative).
pub struct Process {
    pub pid: pid_t
}

enum Req {
    NewChild(libc::pid_t, Sender<ProcessExit>, u64),
}

const CLOEXEC_MSG_FOOTER: &'static [u8] = b"NOEX";

impl Process {
    pub fn id(&self) -> pid_t {
        self.pid
    }

    pub unsafe fn kill(&self, signal: int) -> IoResult<()> {
        Process::killpid(self.pid, signal)
    }

    pub unsafe fn killpid(pid: pid_t, signal: int) -> IoResult<()> {
        let r = libc::funcs::posix88::signal::kill(pid, signal as c_int);
        mkerr_libc(r)
    }

    #[cfg(stage0)]
    pub fn spawn<K, V, C, P>(cfg: &C, in_fd: Option<P>,
                              out_fd: Option<P>, err_fd: Option<P>)
                              -> IoResult<Process>
        where C: ProcessConfig<K, V>, P: AsInner<FileDesc>,
              K: BytesContainer + Eq + Hash<Hasher>, V: BytesContainer
    {
        use libc::funcs::posix88::unistd::{fork, dup2, close, chdir, execvp};

        mod rustrt {
            extern {
                pub fn rust_unset_sigprocmask();
            }
        }

        #[cfg(all(target_os = "android", target_arch = "aarch64"))]
        unsafe fn getdtablesize() -> c_int {
            libc::sysconf(libc::consts::os::sysconf::_SC_OPEN_MAX) as c_int
        }
        #[cfg(not(all(target_os = "android", target_arch = "aarch64")))]
        unsafe fn getdtablesize() -> c_int {
            libc::funcs::bsd44::getdtablesize()
        }

        unsafe fn set_cloexec(fd: c_int) {
            let ret = c::ioctl(fd, c::FIOCLEX);
            assert_eq!(ret, 0);
        }

        let dirp = cfg.cwd().map(|c| c.as_ptr()).unwrap_or(ptr::null());

        // temporary until unboxed closures land
        let cfg = unsafe {
            mem::transmute::<&ProcessConfig<K,V>,&'static ProcessConfig<K,V>>(cfg)
        };

        with_envp(cfg.env(), move|envp: *const c_void| {
            with_argv(cfg.program(), cfg.args(), move|argv: *const *const libc::c_char| unsafe {
                let (input, mut output) = try!(sys::os::pipe());

                // We may use this in the child, so perform allocations before the
                // fork
                let devnull = b"/dev/null\0";

                set_cloexec(output.fd());

                let pid = fork();
                if pid < 0 {
                    return Err(super::last_error())
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
                    return match input.read(&mut bytes) {
                        Ok(8) => {
                            assert!(combine(CLOEXEC_MSG_FOOTER) == combine(&bytes[4.. 8]),
                                "Validation on the CLOEXEC pipe failed: {:?}", bytes);
                            let errno = combine(&bytes[0.. 4]);
                            assert!(p.wait(0).is_ok(), "wait(0) should either return Ok or panic");
                            Err(super::decode_error(errno))
                        }
                        Err(ref e) if e.kind == EndOfFile => Ok(p),
                        Err(e) => {
                            assert!(p.wait(0).is_ok(), "wait(0) should either return Ok or panic");
                            panic!("the CLOEXEC pipe failed: {:?}", e)
                        },
                        Ok(..) => { // pipe I/O up to PIPE_BUF bytes should be atomic
                            assert!(p.wait(0).is_ok(), "wait(0) should either return Ok or panic");
                            panic!("short read on the CLOEXEC pipe")
                        }
                    };
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
                let _ = libc::close(input.fd());

                fn fail(output: &mut FileDesc) -> ! {
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

                // If a stdio file descriptor is set to be ignored (via a -1 file
                // descriptor), then we don't actually close it, but rather open
                // up /dev/null into that file descriptor. Otherwise, the first file
                // descriptor opened up in the child would be numbered as one of the
                // stdio file descriptors, which is likely to wreak havoc.
                let setup = |src: Option<P>, dst: c_int| {
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
                            let fd = obj.as_inner().fd();
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
                    if fd != output.fd() {
                        let _ = close(fd as c_int);
                    }
                }

                match cfg.gid() {
                    Some(u) => {
                        if libc::setgid(u as libc::gid_t) != 0 {
                            fail(&mut output);
                        }
                    }
                    None => {}
                }
                match cfg.uid() {
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
                if cfg.detach() {
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
    #[cfg(not(stage0))]
    pub fn spawn<K, V, C, P>(cfg: &C, in_fd: Option<P>,
                              out_fd: Option<P>, err_fd: Option<P>)
                              -> IoResult<Process>
        where C: ProcessConfig<K, V>, P: AsInner<FileDesc>,
              K: BytesContainer + Eq + Hash, V: BytesContainer
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

        let dirp = cfg.cwd().map(|c| c.as_ptr()).unwrap_or(ptr::null());

        // temporary until unboxed closures land
        let cfg = unsafe {
            mem::transmute::<&ProcessConfig<K,V>,&'static ProcessConfig<K,V>>(cfg)
        };

        with_envp(cfg.env(), move|envp: *const c_void| {
            with_argv(cfg.program(), cfg.args(), move|argv: *const *const libc::c_char| unsafe {
                let (input, mut output) = try!(sys::os::pipe());

                // We may use this in the child, so perform allocations before the
                // fork
                let devnull = b"/dev/null\0";

                set_cloexec(output.fd());

                let pid = fork();
                if pid < 0 {
                    return Err(super::last_error())
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
                    return match input.read(&mut bytes) {
                        Ok(8) => {
                            assert!(combine(CLOEXEC_MSG_FOOTER) == combine(&bytes[4.. 8]),
                                "Validation on the CLOEXEC pipe failed: {:?}", bytes);
                            let errno = combine(&bytes[0.. 4]);
                            assert!(p.wait(0).is_ok(), "wait(0) should either return Ok or panic");
                            Err(super::decode_error(errno))
                        }
                        Err(ref e) if e.kind == EndOfFile => Ok(p),
                        Err(e) => {
                            assert!(p.wait(0).is_ok(), "wait(0) should either return Ok or panic");
                            panic!("the CLOEXEC pipe failed: {:?}", e)
                        },
                        Ok(..) => { // pipe I/O up to PIPE_BUF bytes should be atomic
                            assert!(p.wait(0).is_ok(), "wait(0) should either return Ok or panic");
                            panic!("short read on the CLOEXEC pipe")
                        }
                    };
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
                let _ = libc::close(input.fd());

                fn fail(output: &mut FileDesc) -> ! {
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

                // If a stdio file descriptor is set to be ignored (via a -1 file
                // descriptor), then we don't actually close it, but rather open
                // up /dev/null into that file descriptor. Otherwise, the first file
                // descriptor opened up in the child would be numbered as one of the
                // stdio file descriptors, which is likely to wreak havoc.
                let setup = |src: Option<P>, dst: c_int| {
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
                            let fd = obj.as_inner().fd();
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
                    if fd != output.fd() {
                        let _ = close(fd as c_int);
                    }
                }

                match cfg.gid() {
                    Some(u) => {
                        if libc::setgid(u as libc::gid_t) != 0 {
                            fail(&mut output);
                        }
                    }
                    None => {}
                }
                match cfg.uid() {
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
                if cfg.detach() {
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

    pub fn wait(&self, deadline: u64) -> IoResult<ProcessExit> {
        use cmp;
        use sync::mpsc::TryRecvError;

        static mut WRITE_FD: libc::c_int = 0;

        let mut status = 0 as c_int;
        if deadline == 0 {
            return match retry(|| unsafe { c::waitpid(self.pid, &mut status, 0) }) {
                -1 => panic!("unknown waitpid error: {:?}", super::last_error()),
                _ => Ok(translate_status(status)),
            }
        }

        // On unix, wait() and its friends have no timeout parameters, so there is
        // no way to time out a thread in wait(). From some googling and some
        // thinking, it appears that there are a few ways to handle timeouts in
        // wait(), but the only real reasonable one for a multi-threaded program is
        // to listen for SIGCHLD.
        //
        // With this in mind, the waiting mechanism with a timeout barely uses
        // waitpid() at all. There are a few times that waitpid() is invoked with
        // WNOHANG, but otherwise all the necessary blocking is done by waiting for
        // a SIGCHLD to arrive (and that blocking has a timeout). Note, however,
        // that waitpid() is still used to actually reap the child.
        //
        // Signal handling is super tricky in general, and this is no exception. Due
        // to the async nature of SIGCHLD, we use the self-pipe trick to transmit
        // data out of the signal handler to the rest of the application. The first
        // idea would be to have each thread waiting with a timeout to read this
        // output file descriptor, but a write() is akin to a signal(), not a
        // broadcast(), so it would only wake up one thread, and possibly the wrong
        // thread. Hence a helper thread is used.
        //
        // The helper thread here is responsible for farming requests for a
        // waitpid() with a timeout, and then processing all of the wait requests.
        // By guaranteeing that only this helper thread is reading half of the
        // self-pipe, we're sure that we'll never lose a SIGCHLD. This helper thread
        // is also responsible for select() to wait for incoming messages or
        // incoming SIGCHLD messages, along with passing an appropriate timeout to
        // select() to wake things up as necessary.
        //
        // The ordering of the following statements is also very purposeful. First,
        // we must be guaranteed that the helper thread is booted and available to
        // receive SIGCHLD signals, and then we must also ensure that we do a
        // nonblocking waitpid() at least once before we go ask the sigchld helper.
        // This prevents the race where the child exits, we boot the helper, and
        // then we ask for the child's exit status (never seeing a sigchld).
        //
        // The actual communication between the helper thread and this thread is
        // quite simple, just a channel moving data around.

        unsafe { HELPER.boot(register_sigchld, waitpid_helper) }

        match self.try_wait() {
            Some(ret) => return Ok(ret),
            None => {}
        }

        let (tx, rx) = channel();
        unsafe { HELPER.send(NewChild(self.pid, tx, deadline)); }
        return match rx.recv() {
            Ok(e) => Ok(e),
            Err(..) => Err(timeout("wait timed out")),
        };

        // Register a new SIGCHLD handler, returning the reading half of the
        // self-pipe plus the old handler registered (return value of sigaction).
        //
        // Be sure to set up the self-pipe first because as soon as we register a
        // handler we're going to start receiving signals.
        fn register_sigchld() -> (libc::c_int, c::sigaction) {
            unsafe {
                let mut pipes = [0; 2];
                assert_eq!(libc::pipe(pipes.as_mut_ptr()), 0);
                set_nonblocking(pipes[0], true).ok().unwrap();
                set_nonblocking(pipes[1], true).ok().unwrap();
                WRITE_FD = pipes[1];

                let mut old: c::sigaction = mem::zeroed();
                let mut new: c::sigaction = mem::zeroed();
                new.sa_handler = sigchld_handler;
                new.sa_flags = c::SA_NOCLDSTOP;
                assert_eq!(c::sigaction(c::SIGCHLD, &new, &mut old), 0);
                (pipes[0], old)
            }
        }

        // Helper thread for processing SIGCHLD messages
        fn waitpid_helper(input: libc::c_int,
                          messages: Receiver<Req>,
                          (read_fd, old): (libc::c_int, c::sigaction)) {
            set_nonblocking(input, true).ok().unwrap();
            let mut set: c::fd_set = unsafe { mem::zeroed() };
            let mut tv: libc::timeval;
            let mut active = Vec::<(libc::pid_t, Sender<ProcessExit>, u64)>::new();
            let max = cmp::max(input, read_fd) + 1;

            'outer: loop {
                // Figure out the timeout of our syscall-to-happen. If we're waiting
                // for some processes, then they'll have a timeout, otherwise we
                // wait indefinitely for a message to arrive.
                //
                // FIXME: sure would be nice to not have to scan the entire array
                let min = active.iter().map(|a| a.2).enumerate().min_by(|p| {
                    p.1
                });
                let (p, idx) = match min {
                    Some((idx, deadline)) => {
                        let now = sys::timer::now();
                        let ms = if now < deadline {deadline - now} else {0};
                        tv = ms_to_timeval(ms);
                        (&mut tv as *mut _, idx)
                    }
                    None => (ptr::null_mut(), -1),
                };

                // Wait for something to happen
                c::fd_set(&mut set, input);
                c::fd_set(&mut set, read_fd);
                match unsafe { c::select(max, &mut set, ptr::null_mut(),
                                         ptr::null_mut(), p) } {
                    // interrupted, retry
                    -1 if os::errno() == libc::EINTR as i32 => continue,

                    // We read something, break out and process
                    1 | 2 => {}

                    // Timeout, the pending request is removed
                    0 => {
                        drop(active.remove(idx));
                        continue
                    }

                    n => panic!("error in select {:?} ({:?})", os::errno(), n),
                }

                // Process any pending messages
                if drain(input) {
                    loop {
                        match messages.try_recv() {
                            Ok(NewChild(pid, tx, deadline)) => {
                                active.push((pid, tx, deadline));
                            }
                            Err(TryRecvError::Disconnected) => {
                                assert!(active.len() == 0);
                                break 'outer;
                            }
                            Err(TryRecvError::Empty) => break,
                        }
                    }
                }

                // If a child exited (somehow received SIGCHLD), then poll all
                // children to see if any of them exited.
                //
                // We also attempt to be responsible netizens when dealing with
                // SIGCHLD by invoking any previous SIGCHLD handler instead of just
                // ignoring any previous SIGCHLD handler. Note that we don't provide
                // a 1:1 mapping of our handler invocations to the previous handler
                // invocations because we drain the `read_fd` entirely. This is
                // probably OK because the kernel is already allowed to coalesce
                // simultaneous signals, we're just doing some extra coalescing.
                //
                // Another point of note is that this likely runs the signal handler
                // on a different thread than the one that received the signal. I
                // *think* this is ok at this time.
                //
                // The main reason for doing this is to allow stdtest to run native
                // tests as well. Both libgreen and libnative are running around
                // with process timeouts, but libgreen should get there first
                // (currently libuv doesn't handle old signal handlers).
                if drain(read_fd) {
                    let i: uint = unsafe { mem::transmute(old.sa_handler) };
                    if i != 0 {
                        assert!(old.sa_flags & c::SA_SIGINFO == 0);
                        (old.sa_handler)(c::SIGCHLD);
                    }

                    // FIXME: sure would be nice to not have to scan the entire
                    //        array...
                    active.retain(|&(pid, ref tx, _)| {
                        let pr = Process { pid: pid };
                        match pr.try_wait() {
                            Some(msg) => { tx.send(msg).unwrap(); false }
                            None => true,
                        }
                    });
                }
            }

            // Once this helper thread is done, we re-register the old sigchld
            // handler and close our intermediate file descriptors.
            unsafe {
                assert_eq!(c::sigaction(c::SIGCHLD, &old, ptr::null_mut()), 0);
                let _ = libc::close(read_fd);
                let _ = libc::close(WRITE_FD);
                WRITE_FD = -1;
            }
        }

        // Drain all pending data from the file descriptor, returning if any data
        // could be drained. This requires that the file descriptor is in
        // nonblocking mode.
        fn drain(fd: libc::c_int) -> bool {
            let mut ret = false;
            loop {
                let mut buf = [0u8; 1];
                match unsafe {
                    libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void,
                               buf.len() as libc::size_t)
                } {
                    n if n > 0 => { ret = true; }
                    0 => return true,
                    -1 if wouldblock() => return ret,
                    n => panic!("bad read {:?} ({:?})", os::last_os_error(), n),
                }
            }
        }

        // Signal handler for SIGCHLD signals, must be async-signal-safe!
        //
        // This function will write to the writing half of the "self pipe" to wake
        // up the helper thread if it's waiting. Note that this write must be
        // nonblocking because if it blocks and the reader is the thread we
        // interrupted, then we'll deadlock.
        //
        // When writing, if the write returns EWOULDBLOCK then we choose to ignore
        // it. At that point we're guaranteed that there's something in the pipe
        // which will wake up the other end at some point, so we just allow this
        // signal to be coalesced with the pending signals on the pipe.
        extern fn sigchld_handler(_signum: libc::c_int) {
            let msg = 1;
            match unsafe {
                libc::write(WRITE_FD, &msg as *const _ as *const libc::c_void, 1)
            } {
                1 => {}
                -1 if wouldblock() => {} // see above comments
                n => panic!("bad error on write fd: {:?} {:?}", n, os::errno()),
            }
        }
    }

    pub fn try_wait(&self) -> Option<ProcessExit> {
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

fn with_argv<T,F>(prog: &CString, args: &[CString],
                  cb: F)
                  -> T
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

#[cfg(stage0)]
fn with_envp<K,V,T,F>(env: Option<&HashMap<K, V>>,
                      cb: F)
                      -> T
    where F : FnOnce(*const c_void) -> T,
          K : BytesContainer + Eq + Hash<Hasher>,
          V : BytesContainer
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
                kv.push_all(pair.0.container_as_bytes());
                kv.push('=' as u8);
                kv.push_all(pair.1.container_as_bytes());
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
#[cfg(not(stage0))]
fn with_envp<K,V,T,F>(env: Option<&HashMap<K, V>>,
                      cb: F)
                      -> T
    where F : FnOnce(*const c_void) -> T,
          K : BytesContainer + Eq + Hash,
          V : BytesContainer
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
                kv.push_all(pair.0.container_as_bytes());
                kv.push('=' as u8);
                kv.push_all(pair.1.container_as_bytes());
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

fn translate_status(status: c_int) -> ProcessExit {
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
        ExitStatus(imp::WEXITSTATUS(status) as int)
    } else {
        ExitSignal(imp::WTERMSIG(status) as int)
    }
}
