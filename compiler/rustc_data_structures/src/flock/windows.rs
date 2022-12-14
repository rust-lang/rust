use std::fs::{File, OpenOptions};
use std::io;
use std::mem;
use std::os::windows::prelude::*;
use std::path::Path;

use winapi::shared::winerror::ERROR_INVALID_FUNCTION;
use winapi::um::fileapi::LockFileEx;
use winapi::um::minwinbase::{LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY, OVERLAPPED};
use winapi::um::winnt::{FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE};

#[derive(Debug)]
pub struct Lock {
    _file: File,
}

impl Lock {
    pub fn new(p: &Path, wait: bool, create: bool, exclusive: bool) -> io::Result<Lock> {
        assert!(
            p.parent().unwrap().exists(),
            "Parent directory of lock-file must exist: {}",
            p.display()
        );

        let share_mode = FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE;

        let mut open_options = OpenOptions::new();
        open_options.read(true).share_mode(share_mode);

        if create {
            open_options.create(true).write(true);
        }

        debug!("attempting to open lock file `{}`", p.display());
        let file = match open_options.open(p) {
            Ok(file) => {
                debug!("lock file opened successfully");
                file
            }
            Err(err) => {
                debug!("error opening lock file: {}", err);
                return Err(err);
            }
        };

        let ret = unsafe {
            let mut overlapped: OVERLAPPED = mem::zeroed();

            let mut dwFlags = 0;
            if !wait {
                dwFlags |= LOCKFILE_FAIL_IMMEDIATELY;
            }

            if exclusive {
                dwFlags |= LOCKFILE_EXCLUSIVE_LOCK;
            }

            debug!("attempting to acquire lock on lock file `{}`", p.display());
            LockFileEx(file.as_raw_handle(), dwFlags, 0, 0xFFFF_FFFF, 0xFFFF_FFFF, &mut overlapped)
        };
        if ret == 0 {
            let err = io::Error::last_os_error();
            debug!("failed acquiring file lock: {}", err);
            Err(err)
        } else {
            debug!("successfully acquired lock");
            Ok(Lock { _file: file })
        }
    }

    pub fn error_unsupported(err: &io::Error) -> bool {
        err.raw_os_error() == Some(ERROR_INVALID_FUNCTION as i32)
    }
}

// Note that we don't need a Drop impl on Windows: The file is unlocked
// automatically when it's closed.
