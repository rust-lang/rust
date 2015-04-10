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
use sys::{handle, retry};
use io;
use libc::{self, c_int, HANDLE};

////////////////////////////////////////////////////////////////////////////////
// Anonymous pipes
////////////////////////////////////////////////////////////////////////////////

struct InnerFd {
    fd: c_int
}

#[derive(Clone)]
pub struct AnonPipe {
    inner: Arc<InnerFd>
}

pub unsafe fn anon_pipe() -> io::Result<(AnonPipe, AnonPipe)> {
    // Windows pipes work subtly differently than unix pipes, and their
    // inheritance has to be handled in a different way that I do not
    // fully understand. Here we explicitly make the pipe non-inheritable,
    // which means to pass it to a subprocess they need to be duplicated
    // first, as in std::run.
    let mut fds = [0; 2];
    match libc::pipe(fds.as_mut_ptr(), 1024 as ::libc::c_uint,
    (libc::O_BINARY | libc::O_NOINHERIT) as c_int) {
        0 => {
            assert!(fds[0] != -1 && fds[0] != 0);
            assert!(fds[1] != -1 && fds[1] != 0);

            Ok((AnonPipe::from_fd(fds[0]), AnonPipe::from_fd(fds[1])))
        }
        _ => Err(io::Error::last_os_error()),
    }
}

impl AnonPipe {
    pub fn from_fd(fd: libc::c_int) -> AnonPipe {
        AnonPipe { inner: Arc::new(InnerFd { fd: fd }) }
    }

    pub fn clone_fd(fd: libc::c_int) -> io::Result<AnonPipe> {
        unsafe {
            let fd = retry(|| libc::dup(fd));
            if fd != -1 {
                Ok(AnonPipe::from_fd(fd))
            } else {
                Err(io::Error::last_os_error())
            }
        }
    }

    pub fn raw(&self) -> HANDLE {
        unsafe { libc::get_osfhandle(self.inner.fd) as libc::HANDLE }
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        handle::read(self.raw(), buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        handle::write(self.raw(), buf)
    }
}

impl Drop for InnerFd {
    fn drop(&mut self) {
        // closing stdio file handles makes no sense, so never do it. Also, note
        // that errors are ignored when closing a file descriptor. The reason
        // for this is that if an error occurs we don't actually know if the
        // file descriptor was closed or not, and if we retried (for something
        // like EINTR), we might close another valid file descriptor (opened
        // after we closed ours.
        if self.fd > libc::STDERR_FILENO {
            let n = unsafe { libc::close(self.fd) };
            if n != 0 {
                println!("error {} when closing file descriptor {}", n, self.fd);
            }
        }
    }
}
