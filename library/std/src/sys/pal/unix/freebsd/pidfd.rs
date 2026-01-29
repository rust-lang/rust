//! [`procdesc(4)`] support for FreeBSD
//!
//! `procdesc` is a file-descriptor-oriented interface to process signalling and control.  It's
//! been available since FreeBSD 9.0, which is older than the oldest version of FreeBSD ever
//! supported by Rust, so there is no need for backwards-compatibility shims.
//!
//! Compared to Linux's process descriptors, there are a few differences:
//!
//! Feature        | FreeBSD                       | Linux
//! ---------------+-------------------------------+---------------------------------------------
//! Creation       | pdfork()                      | Any process-creation syscall plus pidfd_open
//! Monitoring     | kevent(), poll(), select()    | epoll(), poll(), select()
//! Convert to pid | pdgetpid()                    | ioctl(PIDFD_GET_INFO)
//! Signalling     | pdkill()                      | pidfd_send_signal
//! Waiting        | Any wait() variant, with pid  | waitid(P_PIDFD)
//! Reaping        | Any wait() variant, or close()| Any wait() variant
//! ---------------+-------------------------------+---------------------------------------------
//!
//! [`procdesc(4)`]: https://man.freebsd.org/cgi/man.cgi?query=procdesc
use crate::io;
use crate::os::fd::{AsRawFd, FromRawFd, IntoRawFd, RawFd};
use crate::sys::fd::FileDesc;
use crate::sys::process::ExitStatus;
use crate::sys::{AsInner, FromInner, IntoInner, cvt};

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub(crate) struct PidFd(FileDesc);

impl PidFd {
    pub fn kill(&self) -> io::Result<()> {
        self.send_signal(libc::SIGKILL)
    }

    pub(crate) fn pid(&self) -> io::Result<u32> {
        let mut pid = 0;
        cvt(unsafe { libc::pdgetpid(self.0.as_raw_fd(), &mut pid) })?;
        Ok(pid as u32)
    }

    fn waitid(&self, options: libc::c_int) -> io::Result<Option<ExitStatus>> {
        // FreeBSD's wait(2) family of functions doesn't yet work with process descriptors
        // directly.  Using waitpid and pdgetpid in combination is technically racy, because the
        // pid might get recycled after waitpid and before dropping the PidFd.  That will be fixed
        // in FreeBSD 16.0.
        //
        // A race-free method for older releases of FreeBSD would be to use kevent with
        // EVFILT_PROCDESC to get the process's exit status, and then close the process descriptor
        // to reap it.  But closing the process descriptor would require a &mut self reference,
        // which this API does not allow.
        //
        // The process descriptor will eventually be closed by OwnedFd::drop .
        let mut siginfo: libc::siginfo_t = unsafe { crate::mem::zeroed() };
        cvt(unsafe {
            libc::waitid(libc::P_PID, self.pid()? as libc::id_t, &mut siginfo, options)
        })?;
        if unsafe { siginfo.si_pid() } == 0 {
            Ok(None)
        } else {
            Ok(Some(ExitStatus::from_waitid_siginfo(siginfo)))
        }
    }

    pub(crate) fn send_signal(&self, signal: i32) -> io::Result<()> {
        cvt(unsafe { libc::pdkill(self.0.as_raw_fd(), signal) }).map(drop)
    }

    pub fn wait(&self) -> io::Result<ExitStatus> {
        let r = self.waitid(libc::WEXITED)?;
        match r {
            Some(exit_status) => Ok(exit_status),
            None => unreachable!("waitid with WEXITED should not return None"),
        }
    }

    pub fn try_wait(&self) -> io::Result<Option<ExitStatus>> {
        self.waitid(libc::WEXITED | libc::WNOHANG)
    }
}

impl AsInner<FileDesc> for PidFd {
    fn as_inner(&self) -> &FileDesc {
        &self.0
    }
}

impl IntoInner<FileDesc> for PidFd {
    fn into_inner(self) -> FileDesc {
        self.0
    }
}

impl FromInner<FileDesc> for PidFd {
    fn from_inner(inner: FileDesc) -> Self {
        Self(inner)
    }
}

impl FromRawFd for PidFd {
    unsafe fn from_raw_fd(fd: RawFd) -> Self {
        Self(FileDesc::from_raw_fd(fd))
    }
}

impl IntoRawFd for PidFd {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_raw_fd()
    }
}
