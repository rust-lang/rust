//! Simple file-locking apis for each OS.
//!
//! This is not meant to be in the standard library, it does nothing with
//! green/native threading. This is just a bare-bones enough solution for
//! librustdoc, it is not production quality at all.

#![allow(non_camel_case_types)]
#![allow(nonstandard_style)]

use std::io;
use std::path::Path;

cfg_if! {
    if #[cfg(unix)] {
        use std::os::unix::prelude::*;
        use std::fs::{File, OpenOptions};

        #[derive(Debug)]
        pub struct Lock {
            _file: File,
        }

        impl Lock {
            pub fn new(p: &Path,
                       wait: bool,
                       create: bool,
                       exclusive: bool)
                       -> io::Result<Lock> {
                let file = OpenOptions::new()
                           .read(true)
                           .write(true)
                           .create(create)
                           .mode(libc::S_IRWXU as u32)
                           .open(p)?;

                let mut operation = if exclusive {
                    libc::LOCK_EX
                } else {
                    libc::LOCK_SH
                };
                if !wait {
                    operation |= libc::LOCK_NB
                }

                let ret = unsafe { libc::flock(file.as_raw_fd(), operation) };
                if ret == -1 {
                    let err = io::Error::last_os_error();
                    Err(err)
                } else {
                    Ok(Lock { _file: file })
                }
            }
        }

        // Note that we don't need a Drop impl to execute `flock(fd, LOCK_UN)`. Lock acquired by
        // `flock` is associated with the file descriptor and closing the file release it
        // automatically.
    } else if #[cfg(windows)] {
        use std::mem;
        use std::os::windows::prelude::*;
        use std::fs::{File, OpenOptions};

        use winapi::um::minwinbase::{OVERLAPPED, LOCKFILE_FAIL_IMMEDIATELY, LOCKFILE_EXCLUSIVE_LOCK};
        use winapi::um::fileapi::LockFileEx;
        use winapi::um::winnt::{FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE};

        #[derive(Debug)]
        pub struct Lock {
            _file: File,
        }

        impl Lock {
            pub fn new(p: &Path,
                       wait: bool,
                       create: bool,
                       exclusive: bool)
                       -> io::Result<Lock> {
                assert!(p.parent().unwrap().exists(),
                    "Parent directory of lock-file must exist: {}",
                    p.display());

                let share_mode = FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE;

                let mut open_options = OpenOptions::new();
                open_options.read(true)
                            .share_mode(share_mode);

                if create {
                    open_options.create(true)
                                .write(true);
                }

                debug!("attempting to open lock file `{}`", p.display());
                let file = match open_options.open(p) {
                    Ok(file) => {
                        debug!("lock file opened successfully");
                        file
                    }
                    Err(err) => {
                        debug!("error opening lock file: {}", err);
                        return Err(err)
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

                    debug!("attempting to acquire lock on lock file `{}`",
                           p.display());
                    LockFileEx(file.as_raw_handle(),
                               dwFlags,
                               0,
                               0xFFFF_FFFF,
                               0xFFFF_FFFF,
                               &mut overlapped)
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
        }

        // Note that we don't need a Drop impl on the Windows: The file is unlocked
        // automatically when it's closed.
    } else {
        #[derive(Debug)]
        pub struct Lock(());

        impl Lock {
            pub fn new(_p: &Path, _wait: bool, _create: bool, _exclusive: bool)
                -> io::Result<Lock>
            {
                let msg = "file locks not supported on this platform";
                Err(io::Error::new(io::ErrorKind::Other, msg))
            }
        }
    }
}
