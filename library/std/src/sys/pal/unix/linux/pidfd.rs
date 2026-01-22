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

    #[cfg(any(test, target_env = "gnu", target_env = "musl"))]
    pub(crate) fn current_process() -> io::Result<PidFd> {
        let pid = crate::process::id();
        let pidfd = cvt(unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) })?;
        Ok(unsafe { PidFd::from_raw_fd(pidfd as RawFd) })
    }

    #[cfg(any(test, target_env = "gnu", target_env = "musl"))]
    pub(crate) fn pid(&self) -> io::Result<u32> {
        use crate::sys::weak::weak;

        // since kernel 6.13
        // https://lore.kernel.org/all/20241010155401.2268522-1-luca.boccassi@gmail.com/
        let mut pidfd_info: libc::pidfd_info = unsafe { crate::mem::zeroed() };
        pidfd_info.mask = libc::PIDFD_INFO_PID as u64;
        match cvt(unsafe { libc::ioctl(self.0.as_raw_fd(), libc::PIDFD_GET_INFO, &mut pidfd_info) })
        {
            Ok(_) => {}
            Err(e) if matches!(e.raw_os_error(), Some(libc::EINVAL | libc::ENOTTY)) => {
                // kernel doesn't support that ioctl, try the glibc helper that looks at procfs
                weak!(
                    fn pidfd_getpid(pidfd: RawFd) -> libc::pid_t;
                );
                if let Some(pidfd_getpid) = pidfd_getpid.get() {
                    let pid: libc::c_int = cvt(unsafe { pidfd_getpid(self.0.as_raw_fd()) })?;
                    return Ok(pid as u32);
                }
                return Err(e);
            }
            Err(e) => return Err(e),
        }

        Ok(pidfd_info.pid)
    }

    fn exit_for_reaped_child(&self) -> io::Result<ExitStatus> {
        // since kernel 6.15
        // https://lore.kernel.org/linux-fsdevel/20250305-work-pidfs-kill_on_last_close-v3-0-c8c3d8361705@kernel.org/T/
        let mut pidfd_info: libc::pidfd_info = unsafe { crate::mem::zeroed() };
        pidfd_info.mask = libc::PIDFD_INFO_EXIT as u64;
        cvt(unsafe { libc::ioctl(self.0.as_raw_fd(), libc::PIDFD_GET_INFO, &mut pidfd_info) })?;
        Ok(ExitStatus::new(pidfd_info.exit_code))
    }

    fn waitid(&self, options: libc::c_int) -> io::Result<Option<ExitStatus>> {
        let mut siginfo: libc::siginfo_t = unsafe { crate::mem::zeroed() };
        let r = cvt(unsafe {
            libc::waitid(libc::P_PIDFD, self.0.as_raw_fd() as u32, &mut siginfo, options)
        });
        match r {
            Err(waitid_err) if waitid_err.raw_os_error() == Some(libc::ECHILD) => {
                // already reaped
                match self.exit_for_reaped_child() {
                    Ok(exit_status) => return Ok(Some(exit_status)),
                    Err(_) => return Err(waitid_err),
                }
            }
            Err(e) => return Err(e),
            Ok(_) => {}
        }
        if unsafe { siginfo.si_pid() } == 0 {
            Ok(None)
        } else {
            Ok(Some(ExitStatus::from_waitid_siginfo(siginfo)))
        }
    }

    pub(crate) fn send_signal(&self, signal: i32) -> io::Result<()> {
        cvt(unsafe {
            libc::syscall(
                libc::SYS_pidfd_send_signal,
                self.0.as_raw_fd(),
                signal,
                crate::ptr::null::<()>(),
                0,
            )
        })
        .map(drop)
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
