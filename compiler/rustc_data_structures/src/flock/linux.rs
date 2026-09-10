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

        match (wait, exclusive) {
            (true, true) => file.lock()?,
            (true, false) => file.lock_shared()?,
            (false, true) => file.try_lock()?,
            (false, false) => file.try_lock_shared()?,
        };

        Ok(Lock { _file: file })
    }

    pub fn error_unsupported(err: &io::Error) -> bool {
        matches!(err.raw_os_error(), Some(libc::ENOTSUP) | Some(libc::ENOSYS))
    }
}

// Note that we don't need a Drop impl to execute `flock(fd, LOCK_UN)`. A lock acquired by
// `flock` is associated with the file descriptor and closing the file releases it
// automatically.
