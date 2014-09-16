// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc;
use std::cmp;
use std::mem;
use std::os;
use std::ptr;
use std::rt::rtio::{IoResult, IoError};

use super::c;
use super::net;
use super::{retry, last_error};

#[deriving(Show)]
pub enum SocketStatus {
    Readable,
    Writable,
}

pub fn timeout(desc: &'static str) -> IoError {
    #[cfg(unix)] use libc::ETIMEDOUT as ERROR;
    #[cfg(windows)] use libc::ERROR_OPERATION_ABORTED as ERROR;
    IoError {
        code: ERROR as uint,
        extra: 0,
        detail: Some(desc.to_string()),
    }
}

pub fn short_write(n: uint, desc: &'static str) -> IoError {
    #[cfg(unix)] use libc::EAGAIN as ERROR;
    #[cfg(windows)] use libc::ERROR_OPERATION_ABORTED as ERROR;
    IoError {
        code: ERROR as uint,
        extra: n,
        detail: Some(desc.to_string()),
    }
}

pub fn eof() -> IoError {
    IoError {
        code: libc::EOF as uint,
        extra: 0,
        detail: None,
    }
}

#[cfg(windows)]
pub fn ms_to_timeval(ms: u64) -> libc::timeval {
    libc::timeval {
        tv_sec: (ms / 1000) as libc::c_long,
        tv_usec: ((ms % 1000) * 1000) as libc::c_long,
    }
}
#[cfg(not(windows))]
pub fn ms_to_timeval(ms: u64) -> libc::timeval {
    libc::timeval {
        tv_sec: (ms / 1000) as libc::time_t,
        tv_usec: ((ms % 1000) * 1000) as libc::suseconds_t,
    }
}

#[cfg(unix)]
pub fn wouldblock() -> bool {
    let err = os::errno();
    err == libc::EWOULDBLOCK as int || err == libc::EAGAIN as int
}

#[cfg(windows)]
pub fn wouldblock() -> bool {
    let err = os::errno();
    err == libc::WSAEWOULDBLOCK as uint
}

#[cfg(unix)]
pub fn set_nonblocking(fd: net::sock_t, nb: bool) -> IoResult<()> {
    let set = nb as libc::c_int;
    super::mkerr_libc(retry(|| unsafe { c::ioctl(fd, c::FIONBIO, &set) }))
}

#[cfg(windows)]
pub fn set_nonblocking(fd: net::sock_t, nb: bool) -> IoResult<()> {
    let mut set = nb as libc::c_ulong;
    if unsafe { c::ioctlsocket(fd, c::FIONBIO, &mut set) != 0 } {
        Err(last_error())
    } else {
        Ok(())
    }
}

// See http://developerweb.net/viewtopic.php?id=3196 for where this is
// derived from.
pub fn connect_timeout(fd: net::sock_t,
                       addrp: *const libc::sockaddr,
                       len: libc::socklen_t,
                       timeout_ms: u64) -> IoResult<()> {
    use std::os;
    #[cfg(unix)]    use libc::EINPROGRESS as INPROGRESS;
    #[cfg(windows)] use libc::WSAEINPROGRESS as INPROGRESS;
    #[cfg(unix)]    use libc::EWOULDBLOCK as WOULDBLOCK;
    #[cfg(windows)] use libc::WSAEWOULDBLOCK as WOULDBLOCK;

    // Make sure the call to connect() doesn't block
    try!(set_nonblocking(fd, true));

    let ret = match unsafe { libc::connect(fd, addrp, len) } {
        // If the connection is in progress, then we need to wait for it to
        // finish (with a timeout). The current strategy for doing this is
        // to use select() with a timeout.
        -1 if os::errno() as int == INPROGRESS as int ||
              os::errno() as int == WOULDBLOCK as int => {
            let mut set: c::fd_set = unsafe { mem::zeroed() };
            c::fd_set(&mut set, fd);
            match await(fd, &mut set, timeout_ms) {
                0 => Err(timeout("connection timed out")),
                -1 => Err(last_error()),
                _ => {
                    let err: libc::c_int = try!(
                        net::getsockopt(fd, libc::SOL_SOCKET, libc::SO_ERROR));
                    if err == 0 {
                        Ok(())
                    } else {
                        Err(IoError {
                            code: err as uint,
                            extra: 0,
                            detail: Some(os::error_string(err as uint)),
                        })
                    }
                }
            }
        }

        -1 => Err(last_error()),
        _ => Ok(()),
    };

    // be sure to turn blocking I/O back on
    try!(set_nonblocking(fd, false));
    return ret;

    #[cfg(unix)]
    fn await(fd: net::sock_t, set: &mut c::fd_set,
             timeout: u64) -> libc::c_int {
        let start = ::io::timer::now();
        retry(|| unsafe {
            // Recalculate the timeout each iteration (it is generally
            // undefined what the value of the 'tv' is after select
            // returns EINTR).
            let mut tv = ms_to_timeval(timeout - (::io::timer::now() - start));
            c::select(fd + 1, ptr::null_mut(), set as *mut _,
                      ptr::null_mut(), &mut tv)
        })
    }
    #[cfg(windows)]
    fn await(_fd: net::sock_t, set: &mut c::fd_set,
             timeout: u64) -> libc::c_int {
        let mut tv = ms_to_timeval(timeout);
        unsafe { c::select(1, ptr::null_mut(), set, ptr::null_mut(), &mut tv) }
    }
}

pub fn await(fds: &[net::sock_t], deadline: Option<u64>,
             status: SocketStatus) -> IoResult<()> {
    let mut set: c::fd_set = unsafe { mem::zeroed() };
    let mut max = 0;
    for &fd in fds.iter() {
        c::fd_set(&mut set, fd);
        max = cmp::max(max, fd + 1);
    }
    if cfg!(windows) {
        max = fds.len() as net::sock_t;
    }

    let (read, write) = match status {
        Readable => (&mut set as *mut _, ptr::null_mut()),
        Writable => (ptr::null_mut(), &mut set as *mut _),
    };
    let mut tv: libc::timeval = unsafe { mem::zeroed() };

    match retry(|| {
        let now = ::io::timer::now();
        let tvp = match deadline {
            None => ptr::null_mut(),
            Some(deadline) => {
                // If we're past the deadline, then pass a 0 timeout to
                // select() so we can poll the status
                let ms = if deadline < now {0} else {deadline - now};
                tv = ms_to_timeval(ms);
                &mut tv as *mut _
            }
        };
        let r = unsafe {
            c::select(max as libc::c_int, read, write, ptr::null_mut(), tvp)
        };
        r
    }) {
        -1 => Err(last_error()),
        0 => Err(timeout("timed out")),
        _ => Ok(()),
    }
}
