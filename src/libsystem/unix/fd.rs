// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use error::prelude::*;
use inner::prelude::*;
use io;
use libc::{self, c_int, size_t, c_void};
use core::mem;
use unix::c;

pub struct FileDesc {
    fd: c_int,
}

impl FileDesc {
    pub fn set_cloexec(&self) {
        unsafe {
            let ret = c::ioctl(self.fd, c::FIOCLEX);
            debug_assert_eq!(ret, 0);
        }
    }
}

impl io::Read for FileDesc {
    fn read(&self, buf: &mut [u8]) -> Result<usize> {
        unsafe {
            match libc::read(self.fd, buf.as_mut_ptr() as *mut c_void, buf.len() as size_t) {
                -1 => Error::expect_last_result(),
                len => Ok(len as usize),
            }
        }
    }
}

impl io::Write for FileDesc {
    fn write(&self, buf: &[u8]) -> Result<usize> {
        unsafe {
            match libc::write(self.fd, buf.as_ptr() as *const c_void, buf.len() as size_t) {
                -1 => Error::expect_last_result(),
                len => Ok(len as usize),
            }
        }
    }
}

impl AsInner<c_int> for FileDesc {
    fn as_inner(&self) -> &c_int { &self.fd }
}

impl IntoInner<c_int> for FileDesc {
    /// Extracts the actual filedescriptor without closing it.
    fn into_inner(self) -> c_int {
        let fd = self.fd;
        mem::forget(self);
        fd
    }
}

impl FromInner<c_int> for FileDesc {
    fn from_inner(fd: c_int) -> Self {
        FileDesc { fd: fd }
    }
}

impl Drop for FileDesc {
    fn drop(&mut self) {
        // Note that errors are ignored when closing a file descriptor. The
        // reason for this is that if an error occurs we don't actually know if
        // the file descriptor was closed or not, and if we retried (for
        // something like EINTR), we might close another valid file descriptor
        // (opened after we closed ours.
        let _ = unsafe { libc::close(self.fd) };
    }
}
