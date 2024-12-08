use crate::io;
use crate::os::fd::{AsRawFd, FromRawFd, RawFd};
use crate::sys::cvt;
use crate::sys::pal::unix::fd::FileDesc;
use crate::sys::process::ExitStatus;
use crate::sys_common::{AsInner, FromInner, IntoInner};

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub(crate) struct PidFd(FileDesc);

impl PidFd {
    pub fn kill(&self) -> io::Result<()> {
        cvt(unsafe {
            libc::syscall(
                libc::SYS_pidfd_send_signal,
                self.0.as_raw_fd(),
                libc::SIGKILL,
                crate::ptr::null::<()>(),
                0,
            )
        })
        .map(drop)
    }

    pub fn wait(&self) -> io::Result<ExitStatus> {
        let mut siginfo: libc::siginfo_t = unsafe { crate::mem::zeroed() };
        cvt(unsafe {
            libc::waitid(libc::P_PIDFD, self.0.as_raw_fd() as u32, &mut siginfo, libc::WEXITED)
        })?;
        Ok(ExitStatus::from_waitid_siginfo(siginfo))
    }

    pub fn try_wait(&self) -> io::Result<Option<ExitStatus>> {
        let mut siginfo: libc::siginfo_t = unsafe { crate::mem::zeroed() };

        cvt(unsafe {
            libc::waitid(
                libc::P_PIDFD,
                self.0.as_raw_fd() as u32,
                &mut siginfo,
                libc::WEXITED | libc::WNOHANG,
            )
        })?;
        if unsafe { siginfo.si_pid() } == 0 {
            Ok(None)
        } else {
            Ok(Some(ExitStatus::from_waitid_siginfo(siginfo)))
        }
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
