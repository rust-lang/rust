#![allow(unused, dead_code)]
use crate::fmt;
use crate::io;
use crate::num::NonZeroI32;
use crate::sys::process::process_common::*;
use crate::sys::{unsupported, unsupported_err};
use core::ffi::NonZero_c_int;

use crate::io::ErrorKind;

use libc::{c_int, pid_t};

pub use crate::sys::{cvt, cvt_r, cvt_nz};

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

impl Command {
    pub fn spawn(
        &mut self,
        default: Stdio,
        needs_stdin: bool,
    ) -> io::Result<(Process, StdioPipes)> {
        let envp = self.capture_env();

        if self.saw_nul() {
            return Err(io::const_io_error!(
                ErrorKind::InvalidInput,
                "nul byte found in provided data",
            ));
        }

        let (ours, theirs) = self.setup_io(default, needs_stdin)?;

        let ret = self.posix_spawn(&theirs, envp.as_ref())?;
        Ok((ret, ours))
    }

    pub fn output(&mut self) -> io::Result<(ExitStatus, Vec<u8>, Vec<u8>)> {
        let (proc, pipes) = self.spawn(Stdio::MakePipe, false)?;
        crate::sys_common::process::wait_with_output(proc, pipes)
    }

    pub fn exec(&mut self, default: Stdio) -> io::Error {
        let envp = self.capture_env();

        if self.saw_nul() {
            return io::const_io_error!(ErrorKind::InvalidInput, "nul byte found in provided data",);
        }

        match self.setup_io(default, true) {
            Ok((_, theirs)) => {
                unsafe {
                    let Err(e) = self.do_exec(theirs, envp.as_ref());
                    e
                }
            }
            Err(e) => e,
        }
    }

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

        if let Some(ref cwd) = *self.get_cwd() {
            cvt(libc::chdir(cwd.as_ptr()))?;
        }

        for callback in self.get_closures().iter_mut() {
            callback()?;
        }

        libc::execvp(self.get_program_cstr().as_ptr(), self.get_argv().as_ptr());
        Err(io::Error::last_os_error())
    }

    fn posix_spawn(
        &mut self,
        stdio: &ChildPipes,
        envp: Option<&CStringArray>,
    ) -> io::Result<Process> {
        use crate::mem::MaybeUninit;
        use crate::sys;

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

            if let Some(pgroup) = pgroup {
                flags |= libc::POSIX_SPAWN_SETPGROUP;
                cvt_nz(libc::posix_spawnattr_setpgroup(attrs.0.as_mut_ptr(), pgroup))?;
            }

            cvt_nz(libc::posix_spawnattr_setflags(attrs.0.as_mut_ptr(), flags as _))?;

            // Make sure we synchronize access to the global `environ` resource
            let envp = envp.map(|c| c.as_ptr()).unwrap_or_else(|| libc::environ as *const _);

            let spawn_fn = libc::posix_spawnp;
            cvt_nz(spawn_fn(
                &mut p.pid,
                self.get_program_cstr().as_ptr(),
                file_actions.0.as_ptr(),
                attrs.0.as_ptr(),
                self.get_argv().as_ptr() as *const _,
                envp as *const _,
            ))?;
            Ok(p)
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Processes
////////////////////////////////////////////////////////////////////////////////

pub struct Process {
    pid: pid_t,
    status: Option<ExitStatus>,
}

impl Process {
    unsafe fn new(pid: pid_t, _pidfd: pid_t) -> Self {
        Process { pid, status: None }
    }

    pub fn id(&self) -> u32 {
        self.pid as u32
    }

    pub fn kill(&mut self) -> io::Result<()> {
        // If we've already waited on this process then the pid can be recycled
        // and used for another process, and we probably shouldn't be killing
        // random processes, so just return an error.
        if self.status.is_some() {
            Err(io::const_io_error!(
                ErrorKind::InvalidInput,
                "invalid argument: can't kill an exited process",
            ))
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

#[derive(PartialEq, Eq, Clone, Copy)]
pub struct ExitStatus(c_int);

impl fmt::Debug for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("wasix_wait_status").field(&self.0).finish()
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
        // procsss_unix/tests.rs) will spot it. `ExitStatusError::code` assumes this too.
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
        ExitStatus(a as i32)
    }
}

impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "exit code: {}", self.0)
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitStatusError(NonZero_c_int);

impl Into<ExitStatus> for ExitStatusError {
    fn into(self) -> ExitStatus {
        ExitStatus(self.0.into())
    }
}

impl ExitStatusError {
    pub fn code(self) -> Option<NonZeroI32> {
        ExitStatus(self.0.into()).code().map(|st| st.try_into().unwrap())
    }
}
