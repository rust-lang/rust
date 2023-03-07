//! We use `flock` rather than `fcntl` on Linux, because WSL1 does not support
//! `fcntl`-style advisory locks properly (rust-lang/rust#72157). For other Unix
//! targets we still use `fcntl` because it's more portable than `flock`.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::prelude::*;
use std::path::Path;

#[derive(Debug)]
pub struct Lock {
    _file: File,
}

impl Lock {
    pub fn new(p: &Path, wait: bool, create: bool, exclusive: bool) -> io::Result<Lock> {
        let file = OpenOptions::new().read(true).write(true).create(create).mode(0o600).open(p)?;

        let mut operation = if exclusive { libc::LOCK_EX } else { libc::LOCK_SH };
        if !wait {
            operation |= libc::LOCK_NB
        }

        let ret = unsafe { libc::flock(file.as_raw_fd(), operation) };
        if ret == -1 { Err(io::Error::last_os_error()) } else { Ok(Lock { _file: file }) }
    }

    pub fn error_unsupported(err: &io::Error) -> bool {
        matches!(err.raw_os_error(), Some(libc::ENOTSUP) | Some(libc::ENOSYS))
    }
}

// Note that we don't need a Drop impl to execute `flock(fd, LOCK_UN)`. A lock acquired by
// `flock` is associated with the file descriptor and closing the file releases it
// automatically.
