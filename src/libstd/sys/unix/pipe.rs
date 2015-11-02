// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use sys::inner::*;
use sys::error::{self, Result};
use sys::unix::fd::FileDesc;
use io;
use libc;

////////////////////////////////////////////////////////////////////////////////
// Anonymous pipes
////////////////////////////////////////////////////////////////////////////////

pub struct AnonPipe(FileDesc);

pub fn anon_pipe() -> Result<(AnonPipe, AnonPipe)> {
    let mut fds = [0; 2];
    if unsafe { libc::pipe(fds.as_mut_ptr()) == 0 } {
        Ok((AnonPipe::from_inner(fds[0]), AnonPipe::from_inner(fds[1])))
    } else {
        error::expect_last_result()
    }
}

impl_inner!(AnonPipe(FileDesc(libc::c_int)): AsInner + IntoInner);
impl_inner!(AnonPipe(FileDesc));

impl FromInner<libc::c_int> for AnonPipe {
    fn from_inner(fd: libc::c_int) -> Self {
        let fd = FileDesc::from_inner(fd);
        fd.set_cloexec();
        AnonPipe(fd)
    }
}

impl io::Read for AnonPipe {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf).map_err(From::from)
    }
}

impl io::Write for AnonPipe {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf).map_err(From::from)
    }

    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}
