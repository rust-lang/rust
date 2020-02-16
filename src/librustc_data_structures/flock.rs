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
        use std::ffi::{CString, OsStr};
        use std::mem;
        use std::os::unix::prelude::*;

        #[derive(Debug)]
        pub struct Lock {
            fd: libc::c_int,
        }

        impl Lock {
            pub fn new(p: &Path,
                       wait: bool,
                       create: bool,
                       exclusive: bool)
                       -> io::Result<Lock> {
                let os: &OsStr = p.as_ref();
                let buf = CString::new(os.as_bytes()).unwrap();
                let open_flags = if create {
                    libc::O_RDWR | libc::O_CREAT
                } else {
                    libc::O_RDWR
                };

                let fd = unsafe {
                    libc::open(buf.as_ptr(), open_flags,
                               libc::S_IRWXU as libc::c_int)
                };

                if fd < 0 {
                    return Err(io::Error::last_os_error());
                }

                let lock_type = if exclusive {
                    libc::F_WRLCK
                } else {
                    libc::F_RDLCK
                };

                let mut flock: libc::flock = unsafe { mem::zeroed() };
                flock.l_type = lock_type as libc::c_short;
                flock.l_whence = libc::SEEK_SET as libc::c_short;
                flock.l_start = 0;
                flock.l_len = 0;

                let cmd = if wait { libc::F_SETLKW } else { libc::F_SETLK };
                let ret = unsafe {
                    libc::fcntl(fd, cmd, &flock)
                };
                if ret == -1 {
                    let err = io::Error::last_os_error();
                    unsafe { libc::close(fd); }
                    Err(err)
                } else {
                    Ok(Lock { fd })
                }
            }
        }

        impl Drop for Lock {
            fn drop(&mut self) {
                let mut flock: libc::flock = unsafe { mem::zeroed() };
                flock.l_type = libc::F_UNLCK as libc::c_short;
                flock.l_whence = libc::SEEK_SET as libc::c_short;
                flock.l_start = 0;
                flock.l_len = 0;

                unsafe {
                    libc::fcntl(self.fd, libc::F_SETLK, &flock);
                    libc::close(self.fd);
                }
            }
        }
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
