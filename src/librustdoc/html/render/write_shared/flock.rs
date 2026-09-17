use std::fs::{File, OpenOptions};
use std::io;
#[cfg(unix)]
use std::os::unix::prelude::*;
use std::path::Path;

#[derive(Debug)]
pub(super) enum Lock {
    /// A well behaved lock scoped to a single fd/handle and unlocked when closing it.
    #[doc(hidden)]
    FdLocked { _file: File },
    /// A fallback implementation using the legacy `fcntl(F_SETLK)` which is scoped to
    /// an entire process. This should only be used when `flock()` or equivalent sane
    /// locking mechanism is unsupported by the OS.
    #[doc(hidden)]
    #[cfg(unix)]
    FcntlFallback { _lock: FcntlLock },
}

impl Lock {
    pub(super) fn new(p: &Path) -> io::Result<Lock> {
        let mut open_options = OpenOptions::new();
        open_options.read(true).write(true).create(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            open_options.mode(0o600);
        }

        let file = open_options.open(p)?;

        match file.lock() {
            Ok(()) => Ok(Lock::FdLocked { _file: file }),
            #[cfg(unix)]
            Err(err) if matches!(err.kind(), io::ErrorKind::Unsupported) => {
                Ok(Lock::FcntlFallback { _lock: FcntlLock::new(file)? })
            }
            Err(err) => Err(err),
        }
    }
}

#[derive(Debug)]
#[cfg(unix)]
pub(super) struct FcntlLock {
    file: File,
}

#[cfg(unix)]
impl FcntlLock {
    fn new(file: File) -> io::Result<FcntlLock> {
        let mut flock: libc::flock = unsafe { std::mem::zeroed() };
        #[cfg(not(all(target_os = "hurd", target_arch = "x86")))]
        {
            flock.l_type = libc::F_WRLCK as libc::c_short;
            flock.l_whence = libc::SEEK_SET as libc::c_short;
        }
        #[cfg(all(target_os = "hurd", target_arch = "x86"))]
        {
            flock.l_type = libc::F_WRLCK as libc::c_int;
            flock.l_whence = libc::SEEK_SET as libc::c_int;
        }
        flock.l_start = 0;
        flock.l_len = 0;

        let ret = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETLKW, &flock) };
        if ret == -1 { Err(io::Error::last_os_error()) } else { Ok(FcntlLock { file }) }
    }
}

#[cfg(unix)]
impl Drop for FcntlLock {
    fn drop(&mut self) {
        let mut flock: libc::flock = unsafe { std::mem::zeroed() };
        #[cfg(not(all(target_os = "hurd", target_arch = "x86")))]
        {
            flock.l_type = libc::F_UNLCK as libc::c_short;
            flock.l_whence = libc::SEEK_SET as libc::c_short;
        }
        #[cfg(all(target_os = "hurd", target_arch = "x86"))]
        {
            flock.l_type = libc::F_UNLCK as libc::c_int;
            flock.l_whence = libc::SEEK_SET as libc::c_int;
        }
        flock.l_start = 0;
        flock.l_len = 0;

        unsafe {
            libc::fcntl(self.file.as_raw_fd(), libc::F_SETLK, &flock);
        }
    }
}
