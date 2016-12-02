// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cmp;
use io;
use libc::{self, c_int};
use mem;
use ptr;
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
        let fd0 = FileDesc::new(fds[0]);
        let fd1 = FileDesc::new(fds[1]);
        Ok((AnonPipe::from_fd(fd0)?, AnonPipe::from_fd(fd1)?))
    } else {
        Err(io::Error::last_os_error())
    }
}

impl AnonPipe {
    pub fn from_fd(fd: FileDesc) -> io::Result<AnonPipe> {
        fd.set_cloexec()?;
        Ok(AnonPipe(fd))
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }

    pub fn read_to_end(&self, buf: &mut Vec<u8>) -> io::Result<usize> {
        self.0.read_to_end(buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    pub fn fd(&self) -> &FileDesc { &self.0 }
    pub fn into_fd(self) -> FileDesc { self.0 }
}

pub fn read2(p1: AnonPipe,
             v1: &mut Vec<u8>,
             p2: AnonPipe,
             v2: &mut Vec<u8>) -> io::Result<()> {

    // Set both pipes into nonblocking mode as we're gonna be reading from both
    // in the `select` loop below, and we wouldn't want one to block the other!
    let p1 = p1.into_fd();
    let p2 = p2.into_fd();
    p1.set_nonblocking(true)?;
    p2.set_nonblocking(true)?;

    let max = cmp::max(p1.raw(), p2.raw());
    loop {
        // wait for either pipe to become readable using `select`
        cvt_r(|| unsafe {
            let mut read: libc::fd_set = mem::zeroed();
            libc::FD_SET(p1.raw(), &mut read);
            libc::FD_SET(p2.raw(), &mut read);
            libc::select(max + 1, &mut read, ptr::null_mut(), ptr::null_mut(),
                         ptr::null_mut())
        })?;

        // Read as much as we can from each pipe, ignoring EWOULDBLOCK or
        // EAGAIN. If we hit EOF, then this will happen because the underlying
        // reader will return Ok(0), in which case we'll see `Ok` ourselves. In
        // this case we flip the other fd back into blocking mode and read
        // whatever's leftover on that file descriptor.
        let read = |fd: &FileDesc, dst: &mut Vec<u8>| {
            match fd.read_to_end(dst) {
                Ok(_) => Ok(true),
                Err(e) => {
                    if e.raw_os_error() == Some(libc::EWOULDBLOCK) ||
                       e.raw_os_error() == Some(libc::EAGAIN) {
                        Ok(false)
                    } else {
                        Err(e)
                    }
                }
            }
        };
        if read(&p1, v1)? {
            p2.set_nonblocking(false)?;
            return p2.read_to_end(v2).map(|_| ());
        }
        if read(&p2, v2)? {
            p1.set_nonblocking(false)?;
            return p1.read_to_end(v1).map(|_| ());
        }
    }
}
