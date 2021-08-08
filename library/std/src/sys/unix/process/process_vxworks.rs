use crate::convert::{TryFrom, TryInto};
use crate::fmt;
use crate::io::{self, Error, ErrorKind};
use crate::num::NonZeroI32;
use crate::os::raw::NonZero_c_int;
use crate::sys;
use crate::sys::cvt;
use crate::sys::process::process_common::*;
use crate::sys_common::thread;
use libc::RTP_ID;
use libc::{self, c_char, c_int};

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

impl Command {
    pub fn spawn(
        &mut self,
        default: Stdio,
        needs_stdin: bool,
    ) -> io::Result<(Process, StdioPipes)> {
        use crate::sys::cvt_r;
        let envp = self.capture_env();

        if self.saw_nul() {
            return Err(io::const_io_error!(
                ErrorKind::InvalidInput,
                "nul byte found in provided data",
            ));
        }
        let (ours, theirs) = self.setup_io(default, needs_stdin)?;
        let mut p = Process { pid: 0, status: None };

        unsafe {
            macro_rules! t {
                ($e:expr) => {
                    match $e {
                        Ok(e) => e,
                        Err(e) => return Err(e.into()),
                    }
                };
            }

            let mut orig_stdin = libc::STDIN_FILENO;
            let mut orig_stdout = libc::STDOUT_FILENO;
            let mut orig_stderr = libc::STDERR_FILENO;

            if let Some(fd) = theirs.stdin.fd() {
                orig_stdin = t!(cvt_r(|| libc::dup(libc::STDIN_FILENO)));
                t!(cvt_r(|| libc::dup2(fd, libc::STDIN_FILENO)));
            }
            if let Some(fd) = theirs.stdout.fd() {
                orig_stdout = t!(cvt_r(|| libc::dup(libc::STDOUT_FILENO)));
                t!(cvt_r(|| libc::dup2(fd, libc::STDOUT_FILENO)));
            }
            if let Some(fd) = theirs.stderr.fd() {
                orig_stderr = t!(cvt_r(|| libc::dup(libc::STDERR_FILENO)));
                t!(cvt_r(|| libc::dup2(fd, libc::STDERR_FILENO)));
            }

            if let Some(ref cwd) = *self.get_cwd() {
                t!(cvt(libc::chdir(cwd.as_ptr())));
            }

            // pre_exec closures are ignored on VxWorks
            let _ = self.get_closures();

            let c_envp = envp
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or_else(|| *sys::os::environ() as *const _);
            let stack_size = thread::min_stack();

            // ensure that access to the environment is synchronized
            let _lock = sys::os::env_read_lock();

            let ret = libc::rtpSpawn(
                self.get_program_cstr().as_ptr(),
                self.get_argv().as_ptr() as *mut *const c_char, // argv
                c_envp as *mut *const c_char,
                100 as c_int, // initial priority
                stack_size,   // initial stack size.
                0,            // options
                0,            // task options
            );

            // Because FileDesc was not used, each duplicated file descriptor
            // needs to be closed manually
            if orig_stdin != libc::STDIN_FILENO {
                t!(cvt_r(|| libc::dup2(orig_stdin, libc::STDIN_FILENO)));
                libc::close(orig_stdin);
            }
            if orig_stdout != libc::STDOUT_FILENO {
                t!(cvt_r(|| libc::dup2(orig_stdout, libc::STDOUT_FILENO)));
                libc::close(orig_stdout);
            }
            if orig_stderr != libc::STDERR_FILENO {
                t!(cvt_r(|| libc::dup2(orig_stderr, libc::STDERR_FILENO)));
                libc::close(orig_stderr);
            }

            if ret != libc::RTP_ID_ERROR {
                p.pid = ret;
                Ok((p, ours))
            } else {
                Err(io::Error::last_os_error())
            }
        }
    }

    pub fn exec(&mut self, default: Stdio) -> io::Error {
        let ret = Command::spawn(self, default, false);
        match ret {
            Ok(t) => unsafe {
                let mut status = 0 as c_int;
                libc::waitpid(t.0.pid, &mut status, 0);
                libc::exit(0);
            },
            Err(e) => e,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Processes
////////////////////////////////////////////////////////////////////////////////

/// The unique id of the process (this should never be negative).
pub struct Process {
    pid: RTP_ID,
    status: Option<ExitStatus>,
}

impl Process {
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

/// Unix exit statuses
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitStatus(c_int);

impl ExitStatus {
    pub fn new(status: c_int) -> ExitStatus {
        ExitStatus(status)
    }

    fn exited(&self) -> bool {
        libc::WIFEXITED(self.0)
    }

    pub fn exit_ok(&self) -> Result<(), ExitStatusError> {
        // This assumes that WIFEXITED(status) && WEXITSTATUS==0 corresponds to status==0.  This is
        // true on all actual versions of Unix, is widely assumed, and is specified in SuS
        // https://pubs.opengroup.org/onlinepubs/9699919799/functions/wait.html .  If it is not
        // true for a platform pretending to be Unix, the tests (our doctests, and also
        // procsss_unix/tests.rs) will spot it.  `ExitStatusError::code` assumes this too.
        match NonZero_c_int::try_from(self.0) {
            Ok(failure) => Err(ExitStatusError(failure)),
            Err(_) => Ok(()),
        }
    }

    pub fn code(&self) -> Option<i32> {
        if self.exited() { Some(libc::WEXITSTATUS(self.0)) } else { None }
    }

    pub fn signal(&self) -> Option<i32> {
        if !self.exited() { Some(libc::WTERMSIG(self.0)) } else { None }
    }

    pub fn core_dumped(&self) -> bool {
        // This method is not yet properly implemented on VxWorks
        false
    }

    pub fn stopped_signal(&self) -> Option<i32> {
        if libc::WIFSTOPPED(self.0) { Some(libc::WSTOPSIG(self.0)) } else { None }
    }

    pub fn continued(&self) -> bool {
        // This method is not yet properly implemented on VxWorks
        false
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

impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(code) = self.code() {
            write!(f, "exit code: {}", code)
        } else {
            let signal = self.signal().unwrap();
            write!(f, "signal: {}", signal)
        }
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
