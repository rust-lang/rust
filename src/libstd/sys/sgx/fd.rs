// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use fortanix_sgx_abi::Fd;

use io;
use mem;
use sys_common::AsInner;
use super::abi::usercalls;

#[derive(Debug)]
pub struct FileDesc {
    fd: Fd,
}

impl FileDesc {
    pub fn new(fd: Fd) -> FileDesc {
        FileDesc { fd: fd }
    }

    pub fn raw(&self) -> Fd { self.fd }

    /// Extracts the actual filedescriptor without closing it.
    pub fn into_raw(self) -> Fd {
        let fd = self.fd;
        mem::forget(self);
        fd
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        usercalls::read(self.fd, buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        usercalls::write(self.fd, buf)
    }

    pub fn flush(&self) -> io::Result<()> {
        usercalls::flush(self.fd)
    }
}

impl AsInner<Fd> for FileDesc {
    fn as_inner(&self) -> &Fd { &self.fd }
}

impl Drop for FileDesc {
    fn drop(&mut self) {
        usercalls::close(self.fd)
    }
}
