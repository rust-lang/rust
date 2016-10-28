// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(reason = "not public", issue = "0", feature = "fd")]

use io::{self, Read};
use libc::{self, c_int, c_void};
use mem;
use sys::cvt;
use sys_common::AsInner;
use sys_common::io::read_to_end_uninitialized;

pub struct FileDesc {
    fd: c_int,
}

impl FileDesc {
    pub fn new(fd: c_int) -> FileDesc {
        FileDesc { fd: fd }
    }

    pub fn raw(&self) -> c_int { self.fd }

    /// Extracts the actual filedescriptor without closing it.
    pub fn into_raw(self) -> c_int {
        let fd = self.fd;
        mem::forget(self);
        fd
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            libc::read(self.fd,
                       buf.as_mut_ptr() as *mut c_void,
                       buf.len())
        })?;
        Ok(ret as usize)
    }

    pub fn read_to_end(&self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let mut me = self;
        (&mut me).read_to_end(buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            libc::write(self.fd,
                        buf.as_ptr() as *const c_void,
                        buf.len())
        })?;
        Ok(ret as usize)
    }

    pub fn set_cloexec(&self) -> io::Result<()> {
        unimplemented!();
        /*
        unsafe {
            let previous = cvt(libc::fcntl(self.fd, libc::F_GETFD, 0))?;
            cvt(libc::fcntl(self.fd, libc::F_SETFD, previous | libc::FD_CLOEXEC))?;
            Ok(())
        }
        */
    }

    pub fn set_nonblocking(&self, _nonblocking: bool) -> io::Result<()> {
        unimplemented!();
        /*
        unsafe {
            let previous = cvt(libc::fcntl(self.fd, libc::F_GETFL, 0))?;
            let new = if nonblocking {
                previous | libc::O_NONBLOCK
            } else {
                previous & !libc::O_NONBLOCK
            };
            cvt(libc::fcntl(self.fd, libc::F_SETFL, new))?;
            Ok(())
        }
        */
    }

    pub fn duplicate(&self) -> io::Result<FileDesc> {
        let new_fd = cvt(unsafe { libc::dup(self.fd) })?;
        Ok(FileDesc::new(new_fd))
    }
}

impl<'a> Read for &'a FileDesc {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        (**self).read(buf)
    }

    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        unsafe { read_to_end_uninitialized(self, buf) }
    }
}

impl AsInner<c_int> for FileDesc {
    fn as_inner(&self) -> &c_int { &self.fd }
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
