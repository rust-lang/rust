use crate::fmt;
use crate::io::{self, Error, ErrorKind};
use crate::mem;
use crate::num::NonZeroI32;
use crate::sys;
use crate::sys::cvt;
use crate::sys::process::process_common::*;
use core::ffi::NonZero_c_int;

#[cfg(target_os = "linux")]
use crate::os::linux::process::PidFd;

#[cfg(target_os = "linux")]
use crate::sys::weak::raw_syscall;

#[cfg(any(
    target_os = "macos",
    target_os = "watchos",
    target_os = "tvos",
    target_os = "freebsd",
    all(target_os = "linux", target_env = "gnu"),
    all(target_os = "linux", target_env = "musl"),
    target_os = "nto",
))]
use crate::sys::weak::weak;

#[cfg(target_os = "vxworks")]
use libc::RTP_ID as pid_t;

#[cfg(not(target_os = "vxworks"))]
use libc::{c_int, pid_t};

#[cfg(not(any(
    target_os = "vxworks",
    target_os = "l4re",
    target_os = "tvos",
    target_os = "watchos",
)))]
use libc::{gid_t, uid_t};

cfg_if::cfg_if! {
    if #[cfg(all(target_os = "nto", target_env = "nto71"))] {
        use crate::thread;
        use libc::{c_char, posix_spawn_file_actions_t, posix_spawnattr_t};
        use crate::time::Duration;
        use crate::sync::LazyLock;
        // Get smallest amount of time we can sleep.
        // Return a common value if it cannot be determined.
        fn get_clock_resolution() -> Duration {
            static MIN_DELAY: LazyLock<Duration, fn() -> Duration> = LazyLock::new(|| {
                let mut mindelay = libc::timespec { tv_sec: 0, tv_nsec: 0 };
                if unsafe { libc::clock_getres(libc::CLOCK_MONOTONIC, &mut mindelay) } == 0
                {
                    Duration::from_nanos(mindelay.tv_nsec as u64)
                } else {
                    Duration::from_millis(1)
                }
            });
            *MIN_DELAY
        }
        // Arbitrary minimum sleep duration for retrying fork/spawn
        const MIN_FORKSPAWN_SLEEP: Duration = Duration::from_nanos(1);
        // Maximum duration of sleeping before giving up and returning an error
        const MAX_FORKSPAWN_SLEEP: Duration = Duration::from_millis(1000);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

impl Command {
    pub fn spawn(
        &mut self,
        default: Stdio,
        needs_stdin: bool,
    ) -> io::Result<(Process, StdioPipes)> {
        const CLOEXEC_MSG_FOOTER: [u8; 4] = *b"NOEX";

        let envp = self.capture_env();

        if self.saw_nul() {
            return Err(io::const_io_error!(
                ErrorKind::InvalidInput,
                "nul byte found in provided data",
            ));
        }

        let (ours, theirs) = self.setup_io(default, needs_stdin)?;

        if let Some(ret) = self.posix_spawn(&theirs, envp.as_ref())? {
            return Ok((ret, ours));
        }
        let (input, output) = sys::pipe::anon_pipe()?;

        // Whatever happens after the fork is almost for sure going to touch or
        // look at the environment in one way or another (PATH in `execvp` or
        // accessing the `environ` pointer ourselves). Make sure no other thread
        // is accessing the environment when we do the fork itself.
        //
        // Note that as soon as we're done with the fork there's no need to hold
        // a lock any more because the parent won't do anything and the child is
        // in its own process. Thus the parent drops the lock guard immediately.
        // The child calls `mem::forget` to leak the lock, which is crucial because
        // releasing a lock is not async-signal-safe.
        let env_lock = sys::os::env_read_lock();
        let (pid, pidfd) = unsafe { self.do_fork()? };

        if pid == 0 {
            crate::panic::always_abort();
            mem::forget(env_lock); // avoid non-async-signal-safe unlocking
            drop(input);
            let Err(err) = unsafe { self.do_exec(theirs, envp.as_ref()) };
            let errno = err.raw_os_error().unwrap_or(libc::EINVAL) as u32;
            let errno = errno.to_be_bytes();
            let bytes = [
                errno[0],
                errno[1],
                errno[2],
                errno[3],
                CLOEXEC_MSG_FOOTER[0],
                CLOEXEC_MSG_FOOTER[1],
                CLOEXEC_MSG_FOOTER[2],
                CLOEXEC_MSG_FOOTER[3],
            ];
            // pipe I/O up to PIPE_BUF bytes should be atomic, and then
            // we want to be sure we *don't* run at_exit destructors as
            // we're being torn down regardless
            rtassert!(output.write(&bytes).is_ok());
            unsafe { libc::_exit(1) }
        }

        drop(env_lock);
        drop(output);

        // Safety: We obtained the pidfd from calling `clone3` with
        // `CLONE_PIDFD` so it's valid an otherwise unowned.
        let mut p = unsafe { Process::new(pid, pidfd) };
        let mut bytes = [0; 8];

        // loop to handle EINTR
        loop {
            match input.read(&mut bytes) {
                Ok(0) => return Ok((p, ours)),
                Ok(8) => {
                    let (errno, footer) = bytes.split_at(4);
                    assert_eq!(
                        CLOEXEC_MSG_FOOTER, footer,
                        "Validation on the CLOEXEC pipe failed: {:?}",
                        bytes
                    );
                    let errno = i32::from_be_bytes(errno.try_into().unwrap());
                    assert!(p.wait().is_ok(), "wait() should either return Ok or panic");
                    return Err(Error::from_raw_os_error(errno));
                }
                Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => {
                    assert!(p.wait().is_ok(), "wait() should either return Ok or panic");
                    panic!("the CLOEXEC pipe failed: {e:?}")
                }
                Ok(..) => {
                    // pipe I/O up to PIPE_BUF bytes should be atomic
                    assert!(p.wait().is_ok(), "wait() should either return Ok or panic");
                    panic!("short read on the CLOEXEC pipe")
                }
            }
        }
    }

    pub fn output(&mut self) -> io::Result<(ExitStatus, Vec<u8>, Vec<u8>)> {
        let (proc, pipes) = self.spawn(Stdio::MakePipe, false)?;
        crate::sys_common::process::wait_with_output(proc, pipes)
    }

    // WatchOS and TVOS headers mark the `fork`/`exec*` functions with
    // `__WATCHOS_PROHIBITED __TVOS_PROHIBITED`, and indicate that the
    // `posix_spawn*` functions should be used instead. It isn't entirely clear
    // what `PROHIBITED` means here (e.g. if calls to these functions are
    // allowed to exist in dead code), but it sounds bad, so we go out of our
    // way to avoid that all-together.
    #[cfg(any(target_os = "tvos", target_os = "watchos"))]
    const ERR_APPLE_TV_WATCH_NO_FORK_EXEC: Error = io::const_io_error!(
        ErrorKind::Unsupported,
        "`fork`+`exec`-based process spawning is not supported on this target",
    );

    #[cfg(any(target_os = "tvos", target_os = "watchos"))]
    unsafe fn do_fork(&mut self) -> Result<(pid_t, pid_t), io::Error> {
        return Err(Self::ERR_APPLE_TV_WATCH_NO_FORK_EXEC);
    }

    // Attempts to fork the process. If successful, returns Ok((0, -1))
    // in the child, and Ok((child_pid, -1)) in the parent.
    #[cfg(not(any(
        target_os = "linux",
        target_os = "watchos",
        target_os = "tvos",
        all(target_os = "nto", target_env = "nto71"),
    )))]
    unsafe fn do_fork(&mut self) -> Result<(pid_t, pid_t), io::Error> {
        cvt(libc::fork()).map(|res| (res, -1))
    }

    // On QNX Neutrino, fork can fail with EBADF in case "another thread might have opened
    // or closed a file descriptor while the fork() was occurring".
    // Documentation says "... or try calling fork() again". This is what we do here.
    // See also https://www.qnx.com/developers/docs/7.1/#com.qnx.doc.neutrino.lib_ref/topic/f/fork.html
    #[cfg(all(target_os = "nto", target_env = "nto71"))]
    unsafe fn do_fork(&mut self) -> Result<(pid_t, pid_t), io::Error> {
        use crate::sys::os::errno;

        let mut delay = MIN_FORKSPAWN_SLEEP;

        loop {
            let r = libc::fork();
            if r == -1 as libc::pid_t && errno() as libc::c_int == libc::EBADF {
                if delay < get_clock_resolution() {
                    // We cannot sleep this short (it would be longer).
                    // Yield instead.
                    thread::yield_now();
                } else if delay < MAX_FORKSPAWN_SLEEP {
                    thread::sleep(delay);
                } else {
                    return Err(io::const_io_error!(
                        ErrorKind::WouldBlock,
                        "forking returned EBADF too often",
                    ));
                }
                delay *= 2;
                continue;
            } else {
                return cvt(r).map(|res| (res, -1));
            }
        }
    }

    // Attempts to fork the process. If successful, returns Ok((0, -1))
    // in the child, and Ok((child_pid, child_pidfd)) in the parent.
    #[cfg(target_os = "linux")]
    unsafe fn do_fork(&mut self) -> Result<(pid_t, pid_t), io::Error> {
        use crate::sync::atomic::{AtomicBool, Ordering};

        static HAS_CLONE3: AtomicBool = AtomicBool::new(true);
        const CLONE_PIDFD: u64 = 0x00001000;

        #[repr(C)]
        struct clone_args {
            flags: u64,
            pidfd: u64,
            child_tid: u64,
            parent_tid: u64,
            exit_signal: u64,
            stack: u64,
            stack_size: u64,
            tls: u64,
            set_tid: u64,
            set_tid_size: u64,
            cgroup: u64,
        }

        raw_syscall! {
            fn clone3(cl_args: *mut clone_args, len: libc::size_t) -> libc::c_long
        }

        // Bypassing libc for `clone3` can make further libc calls unsafe,
        // so we use it sparingly for now. See #89522 for details.
        // Some tools (e.g. sandboxing tools) may also expect `fork`
        // rather than `clone3`.
        let want_clone3_pidfd = self.get_create_pidfd();

        // If we fail to create a pidfd for any reason, this will
        // stay as -1, which indicates an error.
        let mut pidfd: pid_t = -1;

        // Attempt to use the `clone3` syscall, which supports more arguments
        // (in particular, the ability to create a pidfd). If this fails,
        // we will fall through this block to a call to `fork()`
        if want_clone3_pidfd && HAS_CLONE3.load(Ordering::Relaxed) {
            let mut args = clone_args {
                flags: CLONE_PIDFD,
                pidfd: &mut pidfd as *mut pid_t as u64,
                child_tid: 0,
                parent_tid: 0,
                exit_signal: libc::SIGCHLD as u64,
                stack: 0,
                stack_size: 0,
                tls: 0,
                set_tid: 0,
                set_tid_size: 0,
                cgroup: 0,
            };

            let args_ptr = &mut args as *mut clone_args;
            let args_size = crate::mem::size_of::<clone_args>();

            let res = cvt(clone3(args_ptr, args_size));
            match res {
                Ok(n) => return Ok((n as pid_t, pidfd)),
                Err(e) => match e.raw_os_error() {
                    // Multiple threads can race to execute this store,
                    // but that's fine - that just means that multiple threads
                    // will have tried and failed to execute the same syscall,
                    // with no other side effects.
                    Some(libc::ENOSYS) => HAS_CLONE3.store(false, Ordering::Relaxed),
                    // Fallback to fork if `EPERM` is returned. (e.g. blocked by seccomp)
                    Some(libc::EPERM) => {}
                    _ => return Err(e),
                },
            }
        }

        // Generally, we just call `fork`. If we get here after wanting `clone3`,
        // then the syscall does not exist or we do not have permission to call it.
        cvt(libc::fork()).map(|res| (res, pidfd))
    }

    pub fn exec(&mut self, default: Stdio) -> io::Error {
        let envp = self.capture_env();

        if self.saw_nul() {
            return io::const_io_error!(ErrorKind::InvalidInput, "nul byte found in provided data",);
        }

        match self.setup_io(default, true) {
            Ok((_, theirs)) => {
                unsafe {
                    // Similar to when forking, we want to ensure that access to
                    // the environment is synchronized, so make sure to grab the
                    // environment lock before we try to exec.
                    let _lock = sys::os::env_read_lock();

                    let Err(e) = self.do_exec(theirs, envp.as_ref());
                    e
                }
            }
            Err(e) => e,
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
    // deadlock on both macOS and FreeBSD. I'm not entirely sure why, but
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
    #[cfg(not(any(target_os = "tvos", target_os = "watchos")))]
    unsafe fn do_exec(
        &mut self,
        stdio: ChildPipes,
        maybe_envp: Option<&CStringArray>,
    ) -> Result<!, io::Error> {
        use crate::sys::{self, cvt_r};

        if let Some(fd) = stdio.stdin.fd() {
            cvt_r(|| libc::dup2(fd, libc::STDIN_FILENO))?;
        }
        if let Some(fd) = stdio.stdout.fd() {
            cvt_r(|| libc::dup2(fd, libc::STDOUT_FILENO))?;
        }
        if let Some(fd) = stdio.stderr.fd() {
            cvt_r(|| libc::dup2(fd, libc::STDERR_FILENO))?;
        }

        #[cfg(not(target_os = "l4re"))]
        {
            if let Some(_g) = self.get_groups() {
                //FIXME: Redox kernel does not support setgroups yet
                #[cfg(not(target_os = "redox"))]
                cvt(libc::setgroups(_g.len().try_into().unwrap(), _g.as_ptr()))?;
            }
            if let Some(u) = self.get_gid() {
                cvt(libc::setgid(u as gid_t))?;
            }
            if let Some(u) = self.get_uid() {
                // When dropping privileges from root, the `setgroups` call
                // will remove any extraneous groups. We only drop groups
                // if the current uid is 0 and we weren't given an explicit
                // set of groups. If we don't call this, then even though our
                // uid has dropped, we may still have groups that enable us to
                // do super-user things.
                //FIXME: Redox kernel does not support setgroups yet
                #[cfg(not(target_os = "redox"))]
                if libc::getuid() == 0 && self.get_groups().is_none() {
                    cvt(libc::setgroups(0, crate::ptr::null()))?;
                }
                cvt(libc::setuid(u as uid_t))?;
            }
        }
        if let Some(ref cwd) = *self.get_cwd() {
            cvt(libc::chdir(cwd.as_ptr()))?;
        }

        if let Some(pgroup) = self.get_pgroup() {
            cvt(libc::setpgid(0, pgroup))?;
        }

        // emscripten has no signal support.
        #[cfg(not(target_os = "emscripten"))]
        {
            // Inherit the signal mask from the parent rather than resetting it (i.e. do not call
            // pthread_sigmask).

            // If #[unix_sigpipe] is specified, don't reset SIGPIPE to SIG_DFL.
            // If #[unix_sigpipe] is not specified, reset SIGPIPE to SIG_DFL for backward compatibility.
            //
            // #[unix_sigpipe] is an opportunity to change the default here.
            if !crate::sys::unix_sigpipe_attr_specified() {
                #[cfg(target_os = "android")] // see issue #88585
                {
                    let mut action: libc::sigaction = mem::zeroed();
                    action.sa_sigaction = libc::SIG_DFL;
                    cvt(libc::sigaction(libc::SIGPIPE, &action, crate::ptr::null_mut()))?;
                }
                #[cfg(not(target_os = "android"))]
                {
                    let ret = sys::signal(libc::SIGPIPE, libc::SIG_DFL);
                    if ret == libc::SIG_ERR {
                        return Err(io::Error::last_os_error());
                    }
                }
            }
        }

        for callback in self.get_closures().iter_mut() {
            callback()?;
        }

        // Although we're performing an exec here we may also return with an
        // error from this function (without actually exec'ing) in which case we
        // want to be sure to restore the global environment back to what it
        // once was, ensuring that our temporary override, when free'd, doesn't
        // corrupt our process's environment.
        let mut _reset = None;
        if let Some(envp) = maybe_envp {
            struct Reset(*const *const libc::c_char);

            impl Drop for Reset {
                fn drop(&mut self) {
                    unsafe {
                        *sys::os::environ() = self.0;
                    }
                }
            }

            _reset = Some(Reset(*sys::os::environ()));
            *sys::os::environ() = envp.as_ptr();
        }

        libc::execvp(self.get_program_cstr().as_ptr(), self.get_argv().as_ptr());
        Err(io::Error::last_os_error())
    }

    #[cfg(any(target_os = "tvos", target_os = "watchos"))]
    unsafe fn do_exec(
        &mut self,
        _stdio: ChildPipes,
        _maybe_envp: Option<&CStringArray>,
    ) -> Result<!, io::Error> {
        return Err(Self::ERR_APPLE_TV_WATCH_NO_FORK_EXEC);
    }

    #[cfg(not(any(
        target_os = "macos",
        target_os = "tvos",
        target_os = "watchos",
        target_os = "freebsd",
        all(target_os = "linux", target_env = "gnu"),
        all(target_os = "linux", target_env = "musl"),
        target_os = "nto",
    )))]
    fn posix_spawn(
        &mut self,
        _: &ChildPipes,
        _: Option<&CStringArray>,
    ) -> io::Result<Option<Process>> {
        Ok(None)
    }

    // Only support platforms for which posix_spawn() can return ENOENT
    // directly.
    #[cfg(any(
        target_os = "macos",
        // FIXME: `target_os = "ios"`?
        target_os = "tvos",
        target_os = "watchos",
        target_os = "freebsd",
        all(target_os = "linux", target_env = "gnu"),
        all(target_os = "linux", target_env = "musl"),
        target_os = "nto",
    ))]
    fn posix_spawn(
        &mut self,
        stdio: &ChildPipes,
        envp: Option<&CStringArray>,
    ) -> io::Result<Option<Process>> {
        use crate::mem::MaybeUninit;
        use crate::sys::{self, cvt_nz, unix_sigpipe_attr_specified};

        if self.get_gid().is_some()
            || self.get_uid().is_some()
            || (self.env_saw_path() && !self.program_is_path())
            || !self.get_closures().is_empty()
            || self.get_groups().is_some()
            || self.get_create_pidfd()
        {
            return Ok(None);
        }

        // Only glibc 2.24+ posix_spawn() supports returning ENOENT directly.
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        {
            if let Some(version) = sys::os::glibc_version() {
                if version < (2, 24) {
                    return Ok(None);
                }
            } else {
                return Ok(None);
            }
        }

        // On QNX Neutrino, posix_spawnp can fail with EBADF in case "another thread might have opened
        // or closed a file descriptor while the posix_spawn() was occurring".
        // Documentation says "... or try calling posix_spawn() again". This is what we do here.
        // See also http://www.qnx.com/developers/docs/7.1/#com.qnx.doc.neutrino.lib_ref/topic/p/posix_spawn.html
        #[cfg(all(target_os = "nto", target_env = "nto71"))]
        unsafe fn retrying_libc_posix_spawnp(
            pid: *mut pid_t,
            file: *const c_char,
            file_actions: *const posix_spawn_file_actions_t,
            attrp: *const posix_spawnattr_t,
            argv: *const *mut c_char,
            envp: *const *mut c_char,
        ) -> io::Result<i32> {
            let mut delay = MIN_FORKSPAWN_SLEEP;
            loop {
                match libc::posix_spawnp(pid, file, file_actions, attrp, argv, envp) {
                    libc::EBADF => {
                        if delay < get_clock_resolution() {
                            // We cannot sleep this short (it would be longer).
                            // Yield instead.
                            thread::yield_now();
                        } else if delay < MAX_FORKSPAWN_SLEEP {
                            thread::sleep(delay);
                        } else {
                            return Err(io::const_io_error!(
                                ErrorKind::WouldBlock,
                                "posix_spawnp returned EBADF too often",
                            ));
                        }
                        delay *= 2;
                        continue;
                    }
                    r => {
                        return Ok(r);
                    }
                }
            }
        }

        // Solaris, glibc 2.29+, and musl 1.24+ can set a new working directory,
        // and maybe others will gain this non-POSIX function too. We'll check
        // for this weak symbol as soon as it's needed, so we can return early
        // otherwise to do a manual chdir before exec.
        weak! {
            fn posix_spawn_file_actions_addchdir_np(
                *mut libc::posix_spawn_file_actions_t,
                *const libc::c_char
            ) -> libc::c_int
        }
        let addchdir = match self.get_cwd() {
            Some(cwd) => {
                if cfg!(any(target_os = "macos", target_os = "tvos", target_os = "watchos")) {
                    // There is a bug in macOS where a relative executable
                    // path like "../myprogram" will cause `posix_spawn` to
                    // successfully launch the program, but erroneously return
                    // ENOENT when used with posix_spawn_file_actions_addchdir_np
                    // which was introduced in macOS 10.15.
                    if self.get_program_kind() == ProgramKind::Relative {
                        return Ok(None);
                    }
                }
                match posix_spawn_file_actions_addchdir_np.get() {
                    Some(f) => Some((f, cwd)),
                    None => return Ok(None),
                }
            }
            None => None,
        };

        let pgroup = self.get_pgroup();

        // Safety: -1 indicates we don't have a pidfd.
        let mut p = unsafe { Process::new(0, -1) };

        struct PosixSpawnFileActions<'a>(&'a mut MaybeUninit<libc::posix_spawn_file_actions_t>);

        impl Drop for PosixSpawnFileActions<'_> {
            fn drop(&mut self) {
                unsafe {
                    libc::posix_spawn_file_actions_destroy(self.0.as_mut_ptr());
                }
            }
        }

        struct PosixSpawnattr<'a>(&'a mut MaybeUninit<libc::posix_spawnattr_t>);

        impl Drop for PosixSpawnattr<'_> {
            fn drop(&mut self) {
                unsafe {
                    libc::posix_spawnattr_destroy(self.0.as_mut_ptr());
                }
            }
        }

        unsafe {
            let mut attrs = MaybeUninit::uninit();
            cvt_nz(libc::posix_spawnattr_init(attrs.as_mut_ptr()))?;
            let attrs = PosixSpawnattr(&mut attrs);

            let mut flags = 0;

            let mut file_actions = MaybeUninit::uninit();
            cvt_nz(libc::posix_spawn_file_actions_init(file_actions.as_mut_ptr()))?;
            let file_actions = PosixSpawnFileActions(&mut file_actions);

            if let Some(fd) = stdio.stdin.fd() {
                cvt_nz(libc::posix_spawn_file_actions_adddup2(
                    file_actions.0.as_mut_ptr(),
                    fd,
                    libc::STDIN_FILENO,
                ))?;
            }
            if let Some(fd) = stdio.stdout.fd() {
                cvt_nz(libc::posix_spawn_file_actions_adddup2(
                    file_actions.0.as_mut_ptr(),
                    fd,
                    libc::STDOUT_FILENO,
                ))?;
            }
            if let Some(fd) = stdio.stderr.fd() {
                cvt_nz(libc::posix_spawn_file_actions_adddup2(
                    file_actions.0.as_mut_ptr(),
                    fd,
                    libc::STDERR_FILENO,
                ))?;
            }
            if let Some((f, cwd)) = addchdir {
                cvt_nz(f(file_actions.0.as_mut_ptr(), cwd.as_ptr()))?;
            }

            if let Some(pgroup) = pgroup {
                flags |= libc::POSIX_SPAWN_SETPGROUP;
                cvt_nz(libc::posix_spawnattr_setpgroup(attrs.0.as_mut_ptr(), pgroup))?;
            }

            // Inherit the signal mask from this process rather than resetting it (i.e. do not call
            // posix_spawnattr_setsigmask).

            // If #[unix_sigpipe] is specified, don't reset SIGPIPE to SIG_DFL.
            // If #[unix_sigpipe] is not specified, reset SIGPIPE to SIG_DFL for backward compatibility.
            //
            // #[unix_sigpipe] is an opportunity to change the default here.
            if !unix_sigpipe_attr_specified() {
                let mut default_set = MaybeUninit::<libc::sigset_t>::uninit();
                cvt(sigemptyset(default_set.as_mut_ptr()))?;
                cvt(sigaddset(default_set.as_mut_ptr(), libc::SIGPIPE))?;
                cvt_nz(libc::posix_spawnattr_setsigdefault(
                    attrs.0.as_mut_ptr(),
                    default_set.as_ptr(),
                ))?;
                flags |= libc::POSIX_SPAWN_SETSIGDEF;
            }

            cvt_nz(libc::posix_spawnattr_setflags(attrs.0.as_mut_ptr(), flags as _))?;

            // Make sure we synchronize access to the global `environ` resource
            let _env_lock = sys::os::env_read_lock();
            let envp = envp.map(|c| c.as_ptr()).unwrap_or_else(|| *sys::os::environ() as *const _);

            #[cfg(not(target_os = "nto"))]
            let spawn_fn = libc::posix_spawnp;
            #[cfg(target_os = "nto")]
            let spawn_fn = retrying_libc_posix_spawnp;

            let spawn_res = spawn_fn(
                &mut p.pid,
                self.get_program_cstr().as_ptr(),
                file_actions.0.as_ptr(),
                attrs.0.as_ptr(),
                self.get_argv().as_ptr() as *const _,
                envp as *const _,
            );

            #[cfg(target_os = "nto")]
            let spawn_res = spawn_res?;

            cvt_nz(spawn_res)?;
            Ok(Some(p))
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Processes
////////////////////////////////////////////////////////////////////////////////

/// The unique ID of the process (this should never be negative).
pub struct Process {
    pid: pid_t,
    status: Option<ExitStatus>,
    // On Linux, stores the pidfd created for this child.
    // This is None if the user did not request pidfd creation,
    // or if the pidfd could not be created for some reason
    // (e.g. the `clone3` syscall was not available).
    #[cfg(target_os = "linux")]
    pidfd: Option<PidFd>,
}

impl Process {
    #[cfg(target_os = "linux")]
    unsafe fn new(pid: pid_t, pidfd: pid_t) -> Self {
        use crate::os::unix::io::FromRawFd;
        use crate::sys_common::FromInner;
        // Safety: If `pidfd` is nonnegative, we assume it's valid and otherwise unowned.
        let pidfd = (pidfd >= 0).then(|| PidFd::from_inner(sys::fd::FileDesc::from_raw_fd(pidfd)));
        Process { pid, status: None, pidfd }
    }

    #[cfg(not(target_os = "linux"))]
    unsafe fn new(pid: pid_t, _pidfd: pid_t) -> Self {
        Process { pid, status: None }
    }

    pub fn id(&self) -> u32 {
        self.pid as u32
    }

    pub fn kill(&mut self) -> io::Result<()> {
        // If we've already waited on this process then the pid can be recycled
        // and used for another process, and we probably shouldn't be killing
        // random processes, so return Ok because the process has exited already.
        if self.status.is_some() {
            Ok(())
        } else {
            cvt(unsafe { libc::kill(self.pid, libc::SIGKILL) }).map(drop)
        }
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        use crate::sys::cvt_r;
        if let Some(status) = self.status {
            return Ok(status);
        }
        let mut status = 0 as c_int;
        cvt_r(|| unsafe { libc::waitpid(self.pid, &mut status, 0) })?;
        self.status = Some(ExitStatus::new(status));
        Ok(ExitStatus::new(status))
    }

    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        if let Some(status) = self.status {
            return Ok(Some(status));
        }
        let mut status = 0 as c_int;
        let pid = cvt(unsafe { libc::waitpid(self.pid, &mut status, libc::WNOHANG) })?;
        if pid == 0 {
            Ok(None)
        } else {
            self.status = Some(ExitStatus::new(status));
            Ok(Some(ExitStatus::new(status)))
        }
    }
}

/// Unix exit statuses
//
// This is not actually an "exit status" in Unix terminology.  Rather, it is a "wait status".
// See the discussion in comments and doc comments for `std::process::ExitStatus`.
#[derive(PartialEq, Eq, Clone, Copy)]
pub struct ExitStatus(c_int);

impl fmt::Debug for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("unix_wait_status").field(&self.0).finish()
    }
}

impl ExitStatus {
    pub fn new(status: c_int) -> ExitStatus {
        ExitStatus(status)
    }

    fn exited(&self) -> bool {
        libc::WIFEXITED(self.0)
    }

    pub fn exit_ok(&self) -> Result<(), ExitStatusError> {
        // This assumes that WIFEXITED(status) && WEXITSTATUS==0 corresponds to status==0. This is
        // true on all actual versions of Unix, is widely assumed, and is specified in SuS
        // https://pubs.opengroup.org/onlinepubs/9699919799/functions/wait.html. If it is not
        // true for a platform pretending to be Unix, the tests (our doctests, and also
        // process_unix/tests.rs) will spot it. `ExitStatusError::code` assumes this too.
        match NonZero_c_int::try_from(self.0) {
            /* was nonzero */ Ok(failure) => Err(ExitStatusError(failure)),
            /* was zero, couldn't convert */ Err(_) => Ok(()),
        }
    }

    pub fn code(&self) -> Option<i32> {
        self.exited().then(|| libc::WEXITSTATUS(self.0))
    }

    pub fn signal(&self) -> Option<i32> {
        libc::WIFSIGNALED(self.0).then(|| libc::WTERMSIG(self.0))
    }

    pub fn core_dumped(&self) -> bool {
        libc::WIFSIGNALED(self.0) && libc::WCOREDUMP(self.0)
    }

    pub fn stopped_signal(&self) -> Option<i32> {
        libc::WIFSTOPPED(self.0).then(|| libc::WSTOPSIG(self.0))
    }

    pub fn continued(&self) -> bool {
        libc::WIFCONTINUED(self.0)
    }

    pub fn into_raw(&self) -> c_int {
        self.0
    }
}

/// Converts a raw `c_int` to a type-safe `ExitStatus` by wrapping it without copying.
impl From<c_int> for ExitStatus {
    fn from(a: c_int) -> ExitStatus {
        ExitStatus(a)
    }
}

/// Convert a signal number to a readable, searchable name.
///
/// This string should be displayed right after the signal number.
/// If a signal is unrecognized, it returns the empty string, so that
/// you just get the number like "0". If it is recognized, you'll get
/// something like "9 (SIGKILL)".
fn signal_string(signal: i32) -> &'static str {
    match signal {
        libc::SIGHUP => " (SIGHUP)",
        libc::SIGINT => " (SIGINT)",
        libc::SIGQUIT => " (SIGQUIT)",
        libc::SIGILL => " (SIGILL)",
        libc::SIGTRAP => " (SIGTRAP)",
        libc::SIGABRT => " (SIGABRT)",
        #[cfg(not(target_os = "l4re"))]
        libc::SIGBUS => " (SIGBUS)",
        libc::SIGFPE => " (SIGFPE)",
        libc::SIGKILL => " (SIGKILL)",
        #[cfg(not(target_os = "l4re"))]
        libc::SIGUSR1 => " (SIGUSR1)",
        libc::SIGSEGV => " (SIGSEGV)",
        #[cfg(not(target_os = "l4re"))]
        libc::SIGUSR2 => " (SIGUSR2)",
        libc::SIGPIPE => " (SIGPIPE)",
        libc::SIGALRM => " (SIGALRM)",
        libc::SIGTERM => " (SIGTERM)",
        #[cfg(not(target_os = "l4re"))]
        libc::SIGCHLD => " (SIGCHLD)",
        #[cfg(not(target_os = "l4re"))]
        libc::SIGCONT => " (SIGCONT)",
        #[cfg(not(target_os = "l4re"))]
        libc::SIGSTOP => " (SIGSTOP)",
        #[cfg(not(target_os = "l4re"))]
        libc::SIGTSTP => " (SIGTSTP)",
        #[cfg(not(target_os = "l4re"))]
        libc::SIGTTIN => " (SIGTTIN)",
        #[cfg(not(target_os = "l4re"))]
        libc::SIGTTOU => " (SIGTTOU)",
        #[cfg(not(target_os = "l4re"))]
        libc::SIGURG => " (SIGURG)",
        #[cfg(not(target_os = "l4re"))]
        libc::SIGXCPU => " (SIGXCPU)",
        #[cfg(not(target_os = "l4re"))]
        libc::SIGXFSZ => " (SIGXFSZ)",
        #[cfg(not(target_os = "l4re"))]
        libc::SIGVTALRM => " (SIGVTALRM)",
        #[cfg(not(target_os = "l4re"))]
        libc::SIGPROF => " (SIGPROF)",
        #[cfg(not(target_os = "l4re"))]
        libc::SIGWINCH => " (SIGWINCH)",
        #[cfg(not(any(target_os = "haiku", target_os = "l4re")))]
        libc::SIGIO => " (SIGIO)",
        #[cfg(target_os = "haiku")]
        libc::SIGPOLL => " (SIGPOLL)",
        #[cfg(not(target_os = "l4re"))]
        libc::SIGSYS => " (SIGSYS)",
        // For information on Linux signals, run `man 7 signal`
        #[cfg(all(
            target_os = "linux",
            any(
                target_arch = "x86_64",
                target_arch = "x86",
                target_arch = "arm",
                target_arch = "aarch64"
            )
        ))]
        libc::SIGSTKFLT => " (SIGSTKFLT)",
        #[cfg(any(target_os = "linux", target_os = "nto"))]
        libc::SIGPWR => " (SIGPWR)",
        #[cfg(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "tvos",
            target_os = "freebsd",
            target_os = "netbsd",
            target_os = "openbsd",
            target_os = "dragonfly",
            target_os = "nto",
        ))]
        libc::SIGEMT => " (SIGEMT)",
        #[cfg(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "tvos",
            target_os = "freebsd",
            target_os = "netbsd",
            target_os = "openbsd",
            target_os = "dragonfly"
        ))]
        libc::SIGINFO => " (SIGINFO)",
        _ => "",
    }
}

impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(code) = self.code() {
            write!(f, "exit status: {code}")
        } else if let Some(signal) = self.signal() {
            let signal_string = signal_string(signal);
            if self.core_dumped() {
                write!(f, "signal: {signal}{signal_string} (core dumped)")
            } else {
                write!(f, "signal: {signal}{signal_string}")
            }
        } else if let Some(signal) = self.stopped_signal() {
            let signal_string = signal_string(signal);
            write!(f, "stopped (not terminated) by signal: {signal}{signal_string}")
        } else if self.continued() {
            write!(f, "continued (WIFCONTINUED)")
        } else {
            write!(f, "unrecognised wait status: {} {:#x}", self.0, self.0)
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub struct ExitStatusError(NonZero_c_int);

impl Into<ExitStatus> for ExitStatusError {
    fn into(self) -> ExitStatus {
        ExitStatus(self.0.into())
    }
}

impl fmt::Debug for ExitStatusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("unix_wait_status").field(&self.0).finish()
    }
}

impl ExitStatusError {
    pub fn code(self) -> Option<NonZeroI32> {
        ExitStatus(self.0.into()).code().map(|st| st.try_into().unwrap())
    }
}

#[cfg(target_os = "linux")]
#[unstable(feature = "linux_pidfd", issue = "82971")]
impl crate::os::linux::process::ChildExt for crate::process::Child {
    fn pidfd(&self) -> io::Result<&PidFd> {
        self.handle
            .pidfd
            .as_ref()
            .ok_or_else(|| Error::new(ErrorKind::Uncategorized, "No pidfd was created."))
    }

    fn take_pidfd(&mut self) -> io::Result<PidFd> {
        self.handle
            .pidfd
            .take()
            .ok_or_else(|| Error::new(ErrorKind::Uncategorized, "No pidfd was created."))
    }
}

#[cfg(test)]
#[path = "process_unix/tests.rs"]
mod tests;
