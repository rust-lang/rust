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
use std::io::IoResult;
use std::io;
use std::mem;
use std::ptr;

use super::c;
use super::net;
use super::{retry, last_error};

pub fn timeout(desc: &'static str) -> io::IoError {
    io::IoError {
        kind: io::TimedOut,
        desc: desc,
        detail: None,
    }
}

pub fn ms_to_timeval(ms: u64) -> libc::timeval {
    libc::timeval {
        tv_sec: (ms / 1000) as libc::time_t,
        tv_usec: ((ms % 1000) * 1000) as libc::suseconds_t,
    }
}

// See http://developerweb.net/viewtopic.php?id=3196 for where this is
// derived from.
pub fn connect_timeout(fd: net::sock_t,
                       addrp: *libc::sockaddr,
                       len: libc::socklen_t,
                       timeout_ms: u64) -> IoResult<()> {
    use std::os;
    #[cfg(unix)]    use INPROGRESS = libc::EINPROGRESS;
    #[cfg(windows)] use INPROGRESS = libc::WSAEINPROGRESS;
    #[cfg(unix)]    use WOULDBLOCK = libc::EWOULDBLOCK;
    #[cfg(windows)] use WOULDBLOCK = libc::WSAEWOULDBLOCK;

    // Make sure the call to connect() doesn't block
    try!(set_nonblocking(fd, true));

    let ret = match unsafe { libc::connect(fd, addrp, len) } {
        // If the connection is in progress, then we need to wait for it to
        // finish (with a timeout). The current strategy for doing this is
        // to use select() with a timeout.
        -1 if os::errno() as int == INPROGRESS as int ||
              os::errno() as int == WOULDBLOCK as int => {
            let mut set: c::fd_set = unsafe { mem::init() };
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
                        Err(io::IoError::from_errno(err as uint, true))
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
    fn set_nonblocking(fd: net::sock_t, nb: bool) -> IoResult<()> {
        let set = nb as libc::c_int;
        super::mkerr_libc(retry(|| unsafe { c::ioctl(fd, c::FIONBIO, &set) }))
    }

    #[cfg(windows)]
    fn set_nonblocking(fd: net::sock_t, nb: bool) -> IoResult<()> {
        let mut set = nb as libc::c_ulong;
        if unsafe { c::ioctlsocket(fd, c::FIONBIO, &mut set) != 0 } {
            Err(last_error())
        } else {
            Ok(())
        }
    }

    #[cfg(unix)]
    fn await(fd: net::sock_t, set: &mut c::fd_set,
             timeout: u64) -> libc::c_int {
        let start = ::io::timer::now();
        retry(|| unsafe {
            // Recalculate the timeout each iteration (it is generally
            // undefined what the value of the 'tv' is after select
            // returns EINTR).
            let tv = ms_to_timeval(timeout - (::io::timer::now() - start));
            c::select(fd + 1, ptr::null(), set as *mut _ as *_,
                      ptr::null(), &tv)
        })
    }
    #[cfg(windows)]
    fn await(_fd: net::sock_t, set: &mut c::fd_set,
             timeout: u64) -> libc::c_int {
        let tv = ms_to_timeval(timeout);
        unsafe { c::select(1, ptr::null(), &*set, ptr::null(), &tv) }
    }
}

pub fn accept_deadline(fd: net::sock_t, deadline: u64) -> IoResult<()> {
    let mut set: c::fd_set = unsafe { mem::init() };
    c::fd_set(&mut set, fd);

    match retry(|| {
        // If we're past the deadline, then pass a 0 timeout to select() so
        // we can poll the status of the socket.
        let now = ::io::timer::now();
        let ms = if deadline < now {0} else {deadline - now};
        let tv = ms_to_timeval(ms);
        let n = if cfg!(windows) {1} else {fd as libc::c_int + 1};
        unsafe { c::select(n, &set, ptr::null(), ptr::null(), &tv) }
    }) {
        -1 => Err(last_error()),
        0 => Err(timeout("accept timed out")),
        _ => return Ok(()),
    }
}
