// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Simple file-locking apis for each OS.
//!
//! This is not meant to be in the standard library, it does nothing with
//! green/native threading. This is just a bare-bones enough solution for
//! librustdoc, it is not production quality at all.

#[allow(non_camel_case_types)];

pub use self::imp::Lock;

#[cfg(unix)]
mod imp {
    use std::libc;

    #[cfg(target_os = "linux")]
    mod os {
        use std::libc;

        pub struct flock {
            l_type: libc::c_short,
            l_whence: libc::c_short,
            l_start: libc::off_t,
            l_len: libc::off_t,
            l_pid: libc::pid_t,

            // not actually here, but brings in line with freebsd
            l_sysid: libc::c_int,
        }

        pub static F_WRLCK: libc::c_short = 1;
        pub static F_UNLCK: libc::c_short = 2;
        pub static F_SETLK: libc::c_int = 6;
        pub static F_SETLKW: libc::c_int = 7;
    }

    #[cfg(target_os = "freebsd")]
    mod os {
        use std::libc;

        pub struct flock {
            l_start: libc::off_t,
            l_len: libc::off_t,
            l_pid: libc::pid_t,
            l_type: libc::c_short,
            l_whence: libc::c_short,
            l_sysid: libc::c_int,
        }

        pub static F_UNLCK: libc::c_short = 2;
        pub static F_WRLCK: libc::c_short = 3;
        pub static F_SETLK: libc::c_int = 12;
        pub static F_SETLKW: libc::c_int = 13;
    }

    #[cfg(target_os = "macos")]
    mod os {
        use std::libc;

        pub struct flock {
            l_start: libc::off_t,
            l_len: libc::off_t,
            l_pid: libc::pid_t,
            l_type: libc::c_short,
            l_whence: libc::c_short,

            // not actually here, but brings in line with freebsd
            l_sysid: libc::c_int,
        }

        pub static F_UNLCK: libc::c_short = 2;
        pub static F_WRLCK: libc::c_short = 3;
        pub static F_SETLK: libc::c_int = 8;
        pub static F_SETLKW: libc::c_int = 9;
    }

    pub struct Lock {
        priv fd: libc::c_int,
    }

    impl Lock {
        pub fn new(p: &Path) -> Lock {
            let fd = p.with_c_str(|s| unsafe {
                libc::open(s, libc::O_RDWR | libc::O_CREAT, libc::S_IRWXU)
            });
            assert!(fd > 0);
            let flock = os::flock {
                l_start: 0,
                l_len: 0,
                l_pid: 0,
                l_whence: libc::SEEK_SET as libc::c_short,
                l_type: os::F_WRLCK,
                l_sysid: 0,
            };
            let ret = unsafe {
                libc::fcntl(fd, os::F_SETLKW, &flock as *os::flock)
            };
            if ret == -1 {
                unsafe { libc::close(fd); }
                fail!("could not lock `{}`", p.display())
            }
            Lock { fd: fd }
        }
    }

    impl Drop for Lock {
        fn drop(&mut self) {
            let flock = os::flock {
                l_start: 0,
                l_len: 0,
                l_pid: 0,
                l_whence: libc::SEEK_SET as libc::c_short,
                l_type: os::F_UNLCK,
                l_sysid: 0,
            };
            unsafe {
                libc::fcntl(self.fd, os::F_SETLK, &flock as *os::flock);
                libc::close(self.fd);
            }
        }
    }
}

#[cfg(windows)]
mod imp {
    use std::libc;
    use std::mem;
    use std::os::win32::as_utf16_p;
    use std::ptr;

    static LOCKFILE_EXCLUSIVE_LOCK: libc::DWORD = 0x00000002;

    extern "system" {
        fn LockFileEx(hFile: libc::HANDLE,
                      dwFlags: libc::DWORD,
                      dwReserved: libc::DWORD,
                      nNumberOfBytesToLockLow: libc::DWORD,
                      nNumberOfBytesToLockHigh: libc::DWORD,
                      lpOverlapped: libc::LPOVERLAPPED) -> libc::BOOL;
        fn UnlockFileEx(hFile: libc::HANDLE,
                        dwReserved: libc::DWORD,
                        nNumberOfBytesToLockLow: libc::DWORD,
                        nNumberOfBytesToLockHigh: libc::DWORD,
                        lpOverlapped: libc::LPOVERLAPPED) -> libc::BOOL;
    }

    pub struct Lock {
        priv handle: libc::HANDLE,
    }

    impl Lock {
        pub fn new(p: &Path) -> Lock {
            let handle = as_utf16_p(p.as_str().unwrap(), |p| unsafe {
                libc::CreateFileW(p, libc::GENERIC_READ, 0, ptr::mut_null(),
                                  libc::CREATE_ALWAYS,
                                  libc::FILE_ATTRIBUTE_NORMAL,
                                  ptr::mut_null())
            });
            assert!(handle as uint != libc::INVALID_HANDLE_VALUE as uint);
            let mut overlapped: libc::OVERLAPPED = unsafe { mem::init() };
            let ret = unsafe {
                LockFileEx(handle, LOCKFILE_EXCLUSIVE_LOCK, 0, 100, 0,
                           &mut overlapped)
            };
            if ret == 0 {
                unsafe { libc::CloseHandle(handle); }
                fail!("could not lock `{}`", p.display())
            }
            Lock { handle: handle }
        }
    }

    impl Drop for Lock {
        fn drop(&mut self) {
            let mut overlapped: libc::OVERLAPPED = unsafe { mem::init() };
            unsafe {
                UnlockFileEx(self.handle, 0, 100, 0, &mut overlapped);
                libc::CloseHandle(self.handle);
            }
        }
    }
}
