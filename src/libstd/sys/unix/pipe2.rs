// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;

use sync::Arc;
use sys::cvt_r;
use sys::fd::FileDesc;
use io;
use libc;

////////////////////////////////////////////////////////////////////////////////
// Anonymous pipes
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct AnonPipe {
    inner: Arc<FileDesc>
}

pub unsafe fn anon_pipe() -> io::Result<(AnonPipe, AnonPipe)> {
    let mut fds = [0; 2];
    if libc::pipe(fds.as_mut_ptr()) == 0 {
        Ok((AnonPipe::from_fd(fds[0]),
            AnonPipe::from_fd(fds[1])))
    } else {
        Err(io::Error::last_os_error())
    }
}

impl AnonPipe {
    pub fn from_fd(fd: libc::c_int) -> AnonPipe {
        AnonPipe { inner: Arc::new(FileDesc::new(fd)) }
    }

    pub fn clone_fd(fd: libc::c_int) -> io::Result<AnonPipe> {
        unsafe { Ok(AnonPipe::from_fd(try!(cvt_r(|| libc::dup(fd))))) }
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    pub fn raw(&self) -> libc::c_int {
        self.inner.raw()
    }
}
