// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
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
use mem;
use sys::{cvt, syscall};
use sys_common::AsInner;
use sys_common::io::read_to_end_uninitialized;

pub struct FileDesc {
    fd: usize,
}

impl FileDesc {
    pub fn new(fd: usize) -> FileDesc {
        FileDesc { fd: fd }
    }

    pub fn raw(&self) -> usize { self.fd }

    /// Extracts the actual filedescriptor without closing it.
    pub fn into_raw(self) -> usize {
        let fd = self.fd;
        mem::forget(self);
        fd
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        cvt(syscall::read(self.fd, buf))
    }

    pub fn read_to_end(&self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let mut me = self;
        (&mut me).read_to_end(buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        cvt(syscall::write(self.fd, buf))
    }

    pub fn duplicate(&self) -> io::Result<FileDesc> {
        let new_fd = cvt(syscall::dup(self.fd, &[]))?;
        Ok(FileDesc::new(new_fd))
    }

    pub fn nonblocking(&self) -> io::Result<bool> {
        let flags = cvt(syscall::fcntl(self.fd, syscall::F_GETFL, 0))?;
        Ok(flags & syscall::O_NONBLOCK == syscall::O_NONBLOCK)
    }

    pub fn set_cloexec(&self) -> io::Result<()> {
        let mut flags = cvt(syscall::fcntl(self.fd, syscall::F_GETFL, 0))?;
        flags |= syscall::O_CLOEXEC;
        cvt(syscall::fcntl(self.fd, syscall::F_SETFL, flags)).and(Ok(()))
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        let mut flags = cvt(syscall::fcntl(self.fd, syscall::F_GETFL, 0))?;
        if nonblocking {
            flags |= syscall::O_NONBLOCK;
        } else {
            flags &= !syscall::O_NONBLOCK;
        }
        cvt(syscall::fcntl(self.fd, syscall::F_SETFL, flags)).and(Ok(()))
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

impl AsInner<usize> for FileDesc {
    fn as_inner(&self) -> &usize { &self.fd }
}

impl Drop for FileDesc {
    fn drop(&mut self) {
        // Note that errors are ignored when closing a file descriptor. The
        // reason for this is that if an error occurs we don't actually know if
        // the file descriptor was closed or not, and if we retried (for
        // something like EINTR), we might close another valid file descriptor
        // (opened after we closed ours.
        let _ = syscall::close(self.fd);
    }
}
