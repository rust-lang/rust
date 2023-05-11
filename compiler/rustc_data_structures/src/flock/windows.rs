use std::fs::{File, OpenOptions};
use std::io;
use std::os::windows::prelude::*;
use std::path::Path;

use windows::{
    Win32::Foundation::{ERROR_INVALID_FUNCTION, HANDLE},
    Win32::Storage::FileSystem::{
        LockFileEx, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, LOCKFILE_EXCLUSIVE_LOCK,
        LOCKFILE_FAIL_IMMEDIATELY, LOCK_FILE_FLAGS,
    },
    Win32::System::IO::OVERLAPPED,
};

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
        open_options.read(true).share_mode(share_mode.0);

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

        let mut flags = LOCK_FILE_FLAGS::default();
        if !wait {
            flags |= LOCKFILE_FAIL_IMMEDIATELY;
        }

        if exclusive {
            flags |= LOCKFILE_EXCLUSIVE_LOCK;
        }

        let mut overlapped = OVERLAPPED::default();

        debug!("attempting to acquire lock on lock file `{}`", p.display());

        unsafe {
            LockFileEx(
                HANDLE(file.as_raw_handle() as isize),
                flags,
                0,
                u32::MAX,
                u32::MAX,
                &mut overlapped,
            )
        }
        .ok()
        .map_err(|e| {
            let err = io::Error::from_raw_os_error(e.code().0);
            debug!("failed acquiring file lock: {}", err);
            err
        })?;

        debug!("successfully acquired lock");
        Ok(Lock { _file: file })
    }

    pub fn error_unsupported(err: &io::Error) -> bool {
        err.raw_os_error() == Some(ERROR_INVALID_FUNCTION.0 as i32)
    }
}

// Note that we don't need a Drop impl on Windows: The file is unlocked
// automatically when it's closed.
