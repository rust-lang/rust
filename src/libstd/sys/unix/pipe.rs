// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io;
use libc::{self, c_int};
use sys::cvt_r;
use sys::fd::FileDesc;

////////////////////////////////////////////////////////////////////////////////
// Anonymous pipes
////////////////////////////////////////////////////////////////////////////////

pub struct AnonPipe(FileDesc);

pub fn anon_pipe() -> io::Result<(AnonPipe, AnonPipe)> {
    let mut fds = [0; 2];

    // Unfortunately the only known way right now to create atomically set the
    // CLOEXEC flag is to use the `pipe2` syscall on Linux. This was added in
    // 2.6.27, however, and because we support 2.6.18 we must detect this
    // support dynamically.
    if cfg!(target_os = "linux") {
        weak! { fn pipe2(*mut c_int, c_int) -> c_int }
        if let Some(pipe) = pipe2.get() {
            match cvt_r(|| unsafe { pipe(fds.as_mut_ptr(), libc::O_CLOEXEC) }) {
                Ok(_) => {
                    return Ok((AnonPipe(FileDesc::new(fds[0])),
                               AnonPipe(FileDesc::new(fds[1]))))
                }
                Err(ref e) if e.raw_os_error() == Some(libc::ENOSYS) => {}
                Err(e) => return Err(e),
            }
        }
    }
    if unsafe { libc::pipe(fds.as_mut_ptr()) == 0 } {
        Ok((AnonPipe::from_fd(fds[0]), AnonPipe::from_fd(fds[1])))
    } else {
        Err(io::Error::last_os_error())
    }
}

impl AnonPipe {
    pub fn from_fd(fd: libc::c_int) -> AnonPipe {
        let fd = FileDesc::new(fd);
        fd.set_cloexec();
        AnonPipe(fd)
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    pub fn fd(&self) -> &FileDesc { &self.0 }
    pub fn into_fd(self) -> FileDesc { self.0 }
}
