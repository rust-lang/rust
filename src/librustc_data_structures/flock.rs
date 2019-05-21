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
        use std::os::unix::prelude::*;

        #[cfg(any(target_os = "linux", target_os = "android"))]
        mod os {
            #[repr(C)]
            pub struct flock {
                pub l_type: libc::c_short,
                pub l_whence: libc::c_short,
                pub l_start: libc::off_t,
                pub l_len: libc::off_t,
                pub l_pid: libc::pid_t,

                // not actually here, but brings in line with freebsd
                pub l_sysid: libc::c_int,
            }
        }

        #[cfg(target_os = "freebsd")]
        mod os {
            #[repr(C)]
            pub struct flock {
                pub l_start: libc::off_t,
                pub l_len: libc::off_t,
                pub l_pid: libc::pid_t,
                pub l_type: libc::c_short,
                pub l_whence: libc::c_short,
                pub l_sysid: libc::c_int,
            }
        }

        #[cfg(any(target_os = "dragonfly",
                  target_os = "netbsd",
                  target_os = "openbsd"))]
        mod os {
            #[repr(C)]
            pub struct flock {
                pub l_start: libc::off_t,
                pub l_len: libc::off_t,
                pub l_pid: libc::pid_t,
                pub l_type: libc::c_short,
                pub l_whence: libc::c_short,

                // not actually here, but brings in line with freebsd
                pub l_sysid: libc::c_int,
            }
        }

        #[cfg(target_os = "haiku")]
        mod os {
            #[repr(C)]
            pub struct flock {
                pub l_type: libc::c_short,
                pub l_whence: libc::c_short,
                pub l_start: libc::off_t,
                pub l_len: libc::off_t,
                pub l_pid: libc::pid_t,

                // not actually here, but brings in line with freebsd
                pub l_sysid: libc::c_int,
            }
        }

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        mod os {
            #[repr(C)]
            pub struct flock {
                pub l_start: libc::off_t,
                pub l_len: libc::off_t,
                pub l_pid: libc::pid_t,
                pub l_type: libc::c_short,
                pub l_whence: libc::c_short,

                // not actually here, but brings in line with freebsd
                pub l_sysid: libc::c_int,
            }
        }

        #[cfg(target_os = "solaris")]
        mod os {
            #[repr(C)]
            pub struct flock {
                pub l_type: libc::c_short,
                pub l_whence: libc::c_short,
                pub l_start: libc::off_t,
                pub l_len: libc::off_t,
                pub l_sysid: libc::c_int,
                pub l_pid: libc::pid_t,
            }
        }

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
                    libc::F_WRLCK as libc::c_short
                } else {
                    libc::F_RDLCK as libc::c_short
                };

                let flock = os::flock {
                    l_start: 0,
                    l_len: 0,
                    l_pid: 0,
                    l_whence: libc::SEEK_SET as libc::c_short,
                    l_type: lock_type,
                    l_sysid: 0,
                };
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
                let flock = os::flock {
                    l_start: 0,
                    l_len: 0,
                    l_pid: 0,
                    l_whence: libc::SEEK_SET as libc::c_short,
                    l_type: libc::F_UNLCK as libc::c_short,
                    l_sysid: 0,
                };
                unsafe {
                    libc::fcntl(self.fd, libc::F_SETLK, &flock);
                    libc::close(self.fd);
                }
            }
        }
    } else if #[cfg(windows)] {
        use std::mem;
        use std::os::windows::prelude::*;
        use std::os::windows::raw::HANDLE;
        use std::fs::{File, OpenOptions};
        use std::os::raw::{c_ulong, c_int};

        type DWORD = c_ulong;
        type BOOL = c_int;
        type ULONG_PTR = usize;

        type LPOVERLAPPED = *mut OVERLAPPED;
        const LOCKFILE_EXCLUSIVE_LOCK: DWORD = 0x0000_0002;
        const LOCKFILE_FAIL_IMMEDIATELY: DWORD = 0x0000_0001;

        const FILE_SHARE_DELETE: DWORD = 0x4;
        const FILE_SHARE_READ: DWORD = 0x1;
        const FILE_SHARE_WRITE: DWORD = 0x2;

        #[repr(C)]
        struct OVERLAPPED {
            Internal: ULONG_PTR,
            InternalHigh: ULONG_PTR,
            Offset: DWORD,
            OffsetHigh: DWORD,
            hEvent: HANDLE,
        }

        extern "system" {
            fn LockFileEx(hFile: HANDLE,
                          dwFlags: DWORD,
                          dwReserved: DWORD,
                          nNumberOfBytesToLockLow: DWORD,
                          nNumberOfBytesToLockHigh: DWORD,
                          lpOverlapped: LPOVERLAPPED) -> BOOL;
        }

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

                debug!("Attempting to open lock file `{}`", p.display());
                let file = match open_options.open(p) {
                    Ok(file) => {
                        debug!("Lock file opened successfully");
                        file
                    }
                    Err(err) => {
                        debug!("Error opening lock file: {}", err);
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

                    debug!("Attempting to acquire lock on lock file `{}`",
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
                    debug!("Failed acquiring file lock: {}", err);
                    Err(err)
                } else {
                    debug!("Successfully acquired lock.");
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

impl Lock {
    pub fn panicking_new(p: &Path,
                         wait: bool,
                         create: bool,
                         exclusive: bool)
                         -> Lock {
        Lock::new(p, wait, create, exclusive).unwrap_or_else(|err| {
            panic!("could not lock `{}`: {}", p.display(), err);
        })
    }
}
