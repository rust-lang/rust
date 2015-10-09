// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use error::prelude::*;
use inner::prelude::*;
use io::prelude::*;

use c_str::CStr;
use core::fmt;
use libc::{self, c_int, size_t};
use core::str;
use core::sync::atomic::{self, AtomicBool};
use unix::c;
use unix::fd::FileDesc;
use common::net::{getsockopt, setsockopt};
use net as sys;
use core::time::Duration;

use unix::cvt_r;

pub type wrlen_t = size_t;

pub struct Socket(FileDesc);

pub fn init() {}

pub fn cvt_gai(err: c_int) -> Result<()> {
    if err == 0 { return Ok(()) }

    Err(Error::from_code(err))
}

impl Socket {
    pub fn new<N: sys::Net>(addr: &sys::SocketAddr<N>, ty: c_int) -> Result<Socket> {
        let fam = match *addr {
            sys::SocketAddr::V4(..) => libc::AF_INET,
            sys::SocketAddr::V6(..) => libc::AF_INET6,
        };
        unsafe {
            let fd = match libc::socket(fam, ty, 0) {
                s if s < 0 => return Error::expect_last_result(),
                s => s,
            };
            let fd = FileDesc::from_inner(fd);
            fd.set_cloexec();
            Ok(Socket(fd))
        }
    }

    pub fn accept(&self, storage: *mut libc::sockaddr,
                  len: *mut libc::socklen_t) -> Result<Socket> {
        let fd = try!(cvt_r(|| unsafe {
            libc::accept(*self.0.as_inner(), storage, len)
        }));
        let fd = FileDesc::from_inner(fd);
        fd.set_cloexec();
        Ok(Socket(fd))
    }

    pub fn duplicate(&self) -> Result<Socket> {
        use libc::funcs::posix88::fcntl::fcntl;
        let make_socket = |fd| {
            let fd = FileDesc::from_inner(fd);
            fd.set_cloexec();
            Socket(fd)
        };
        static EMULATE_F_DUPFD_CLOEXEC: AtomicBool = AtomicBool::new(false);
        if !EMULATE_F_DUPFD_CLOEXEC.load(atomic::Ordering::Relaxed) {
            match unsafe { fcntl(*self.0.as_inner(), libc::F_DUPFD_CLOEXEC, 0) } {
                // `EINVAL` can only be returned on two occasions: Invalid
                // command (second parameter) or invalid third parameter. 0 is
                // always a valid third parameter, so it must be the second
                // parameter.
                //
                // Store the result in a global variable so we don't try each
                // syscall twice.
                e if e < 0 => match Error::expect_last_result() {
                    Err(ref e) if e.code() == libc::EINVAL =>
                        EMULATE_F_DUPFD_CLOEXEC.store(true, atomic::Ordering::Relaxed),
                    e => return e
                },
                e => return Ok(make_socket(e)),
            }
        }
        match unsafe { fcntl(*self.0.as_inner(), libc::F_DUPFD, 0) } {
            e if e < 0 => Error::expect_last_result(),
            e => Ok(make_socket(e)),
        }
    }

    pub fn read(&self, buf: &mut [u8]) -> Result<usize> {
        self.0.read(buf)
    }

    pub fn set_timeout(&self, dur: Option<Duration>, kind: libc::c_int) -> Result<()> {
        let timeout = match dur {
            Some(dur) => {
                if dur.as_secs() == 0 && dur.subsec_nanos() == 0 {
                    return Err(Error::from_code(libc::EINVAL));
                }

                let secs = if dur.as_secs() > libc::time_t::max_value() as u64 {
                    libc::time_t::max_value()
                } else {
                    dur.as_secs() as libc::time_t
                };
                let mut timeout = libc::timeval {
                    tv_sec: secs,
                    tv_usec: (dur.subsec_nanos() / 1000) as libc::suseconds_t,
                };
                if timeout.tv_sec == 0 && timeout.tv_usec == 0 {
                    timeout.tv_usec = 1;
                }
                timeout
            }
            None => {
                libc::timeval {
                    tv_sec: 0,
                    tv_usec: 0,
                }
            }
        };
        setsockopt(self, libc::SOL_SOCKET, kind, timeout)
    }

    pub fn timeout(&self, kind: libc::c_int) -> Result<Option<Duration>> {
        let raw: libc::timeval = try!(getsockopt(self, libc::SOL_SOCKET, kind));
        if raw.tv_sec == 0 && raw.tv_usec == 0 {
            Ok(None)
        } else {
            let sec = raw.tv_sec as u64;
            let nsec = (raw.tv_usec as u32) * 1000;
            Ok(Some(Duration::new(sec, nsec)))
        }
    }
}

impl AsInner<c_int> for Socket {
    fn as_inner(&self) -> &c_int { self.0.as_inner() }
}

impl FromInner<c_int> for Socket {
    fn from_inner(fd: c_int) -> Socket { Socket(FileDesc::from_inner(fd)) }
}

impl IntoInner<c_int> for Socket {
    fn into_inner(self) -> c_int { self.0.into_inner() }
}

impl fmt::Debug for Socket {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self.0.as_inner(), f)
    }
}
