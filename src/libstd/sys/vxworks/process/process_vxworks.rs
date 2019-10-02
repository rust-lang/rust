use crate::io::{self, Error, ErrorKind};
use libc::{self, c_int, c_char};
use libc::{RTP_ID};
use crate::sys;
use crate::sys::cvt;
use crate::sys::process::process_common::*;
use crate::sys_common::thread;

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

impl Command {
    pub fn spawn(&mut self, default: Stdio, needs_stdin: bool)
                 -> io::Result<(Process, StdioPipes)> {
        use crate::sys::{cvt_r};
        const CLOEXEC_MSG_FOOTER: &'static [u8] = b"NOEX";

        if self.saw_nul() {
            return Err(io::Error::new(ErrorKind::InvalidInput,
                                      "nul byte found in provided data"));
        }
        let (ours, theirs) = self.setup_io(default, needs_stdin)?;
        let mut p = Process { pid: 0, status: None };

        unsafe {
            macro_rules! t {
                ($e:expr) => (match $e {
                    Ok(e) => e,
                    Err(e) => return Err(e.into()),
                })
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

            let ret = libc::rtpSpawn(
                self.get_argv()[0],                   // executing program
                self.get_argv().as_ptr() as *mut *const c_char, // argv
                *sys::os::environ() as *mut *const c_char,
                100 as c_int,                         // initial priority
                thread::min_stack(),                  // initial stack size.
                0,                                    // options
                0                                     // task options
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
            Err(Error::new(ErrorKind::InvalidInput,
                           "invalid argument: can't kill an exited process"))
        } else {
            cvt(unsafe { libc::kill(self.pid, libc::SIGKILL) }).map(|_| ())
        }
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        use crate::sys::cvt_r;
        if let Some(status) = self.status {
            return Ok(status)
        }
        let mut status = 0 as c_int;
        cvt_r(|| unsafe { libc::waitpid(self.pid, &mut status, 0) })?;
        self.status = Some(ExitStatus::new(status));
        Ok(ExitStatus::new(status))
    }

    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        if let Some(status) = self.status {
            return Ok(Some(status))
        }
        let mut status = 0 as c_int;
        let pid = cvt(unsafe {
            libc::waitpid(self.pid, &mut status, libc::WNOHANG)
        })?;
        if pid == 0 {
            Ok(None)
        } else {
            self.status = Some(ExitStatus::new(status));
            Ok(Some(ExitStatus::new(status)))
        }
    }
}
