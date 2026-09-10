//! Simple file-locking apis for each OS.
//!
//! This is not meant to be in the standard library, it does nothing with
//! green/native threading. This is just a bare-bones enough solution for
//! librustdoc, it is not production quality at all.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

#[derive(Debug)]
pub enum Lock {
    /// A well behaved lock scoped to a single fd/handle and unlocked when closing it.
    #[doc(hidden)]
    FdLocked { _file: File },
    /// A fallback implementation which may for example be scoped to an entire process,
    /// like legacy `fcntl(F_SETLK)` on Unix. This should only be used when `flock()`
    /// or equivalent sane locking mechanism is unsupported by the OS.
    #[doc(hidden)]
    Fallback(fallback::Lock),
}

impl Lock {
    pub fn new(p: &Path, wait: bool, create: bool, exclusive: bool) -> io::Result<Lock> {
        let mut open_options = OpenOptions::new();
        open_options.read(true).write(true).create(create);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            open_options.mode(0o600);
        }

        let file = open_options.open(p)?;

        let res = match (wait, exclusive) {
            (true, true) => file.lock(),
            (true, false) => file.lock_shared(),
            (false, true) => file.try_lock().map_err(io::Error::from),
            (false, false) => file.try_lock_shared().map_err(io::Error::from),
        };

        match res {
            Ok(()) => Ok(Lock::FdLocked { _file: file }),
            Err(err) if matches!(err.kind(), io::ErrorKind::Unsupported) => {
                Ok(Lock::Fallback(fallback::Lock::new(p, wait, create, exclusive)?))
            }
            Err(err) => Err(err),
        }
    }

    pub fn error_unsupported(err: &io::Error) -> bool {
        #[cfg(windows)]
        if err.raw_os_error() == Some(windows::Win32::Foundation::ERROR_INVALID_FUNCTION.0 as i32) {
            // Not mapped to ErrorKind::Unsupported by libstd
            return true;
        }

        matches!(err.kind(), io::ErrorKind::Unsupported)
    }
}

cfg_select! {
    unix => {
        mod unix;
        use unix as fallback;
    }
    _ => {
        mod unsupported;
        use unsupported as fallback;
    }
}
