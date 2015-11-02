// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use sys::error::Result;
use sys::unix::{c, cvt};
use sys::unix::platform::raw::off_t;
use libc::{self, c_int, size_t, c_void};
use io;

pub struct FileDesc(c_int);

impl FileDesc {
    pub fn read(&self, buf: &mut [u8]) -> Result<usize> {
        cvt(unsafe { libc::read(self.0, buf.as_mut_ptr() as *mut c_void, buf.len() as size_t) })
            .map(|len| len as usize)
    }

    pub fn write(&self, buf: &[u8]) -> Result<usize> {
        cvt(unsafe { libc::write(self.0, buf.as_ptr() as *const c_void, buf.len() as size_t) })
            .map(|len| len as usize)
    }

    pub fn seek(&self, pos: io::SeekFrom) -> Result<u64> {
        let (whence, pos) = match pos {
            io::SeekFrom::Start(off) => (libc::SEEK_SET, off as off_t),
            io::SeekFrom::End(off) => (libc::SEEK_END, off as off_t),
            io::SeekFrom::Current(off) => (libc::SEEK_CUR, off as off_t),
        };
        cvt(unsafe { libc::lseek(self.0, pos, whence) }).map(|pos| pos as u64)
    }

    #[cfg(not(target_env = "newlib"))]
    pub fn set_cloexec(&self) {
        unsafe {
            let ret = c::ioctl(self.0, c::FIOCLEX);
            debug_assert_eq!(ret, 0);
        }
    }

    #[cfg(target_env = "newlib")]
    pub fn set_cloexec(&self) {
        unsafe {
            let previous = c::fnctl(self.0, c::F_GETFD);
            let ret = c::fnctl(self.0, c::F_SETFD, previous | c::FD_CLOEXEC);
            debug_assert_eq!(ret, 0);
        }
    }
}

impl_inner!(FileDesc(c_int): AsInner + IntoInnerForget + FromInner);

impl Drop for FileDesc {
    fn drop(&mut self) {
        // Note that errors are ignored when closing a file descriptor. The
        // reason for this is that if an error occurs we don't actually know if
        // the file descriptor was closed or not, and if we retried (for
        // something like EINTR), we might close another valid file descriptor
        // (opened after we closed ours.
        let _ = unsafe { libc::close(self.0) };
    }
}
