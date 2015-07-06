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

use ffi::CStr;
use io;
use libc::{self, c_int, size_t};
use str;
use sys::c;
use net::SocketAddr;
use sys::fd::FileDesc;
use sys_common::{AsInner, FromInner, IntoInner};
use sys_common::net::{getsockopt, setsockopt};
use time::Duration;

pub use sys::{cvt, cvt_r};

pub type wrlen_t = size_t;

pub struct Socket(FileDesc);

pub fn init() {}

pub fn cvt_gai(err: c_int) -> io::Result<()> {
    if err == 0 { return Ok(()) }

    let detail = unsafe {
        str::from_utf8(CStr::from_ptr(c::gai_strerror(err)).to_bytes()).unwrap()
            .to_string()
    };
    Err(io::Error::new(io::ErrorKind::Other,
                       &format!("failed to lookup address information: {}",
                                detail)[..]))
}

impl Socket {
    pub fn new(addr: &SocketAddr, ty: c_int) -> io::Result<Socket> {
        let fam = match *addr {
            SocketAddr::V4(..) => libc::AF_INET,
            SocketAddr::V6(..) => libc::AF_INET6,
        };
        unsafe {
            let fd = try!(cvt(libc::socket(fam, ty, 0)));
            let fd = FileDesc::new(fd);
            fd.set_cloexec();
            Ok(Socket(fd))
        }
    }

    pub fn accept(&self, storage: *mut libc::sockaddr,
                  len: *mut libc::socklen_t) -> io::Result<Socket> {
        let fd = try!(cvt_r(|| unsafe {
            libc::accept(self.0.raw(), storage, len)
        }));
        let fd = FileDesc::new(fd);
        fd.set_cloexec();
        Ok(Socket(fd))
    }

    pub fn duplicate(&self) -> io::Result<Socket> {
        let fd = try!(cvt(unsafe { libc::dup(self.0.raw()) }));
        let fd = FileDesc::new(fd);
        fd.set_cloexec();
        Ok(Socket(fd))
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }

    pub fn set_timeout(&self, dur: Option<Duration>, kind: libc::c_int) -> io::Result<()> {
        let timeout = match dur {
            Some(dur) => {
                if dur.as_secs() == 0 && dur.subsec_nanos() == 0 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput,
                                              "cannot set a 0 duration timeout"));
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

    pub fn timeout(&self, kind: libc::c_int) -> io::Result<Option<Duration>> {
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
    fn from_inner(fd: c_int) -> Socket { Socket(FileDesc::new(fd)) }
}

impl IntoInner<c_int> for Socket {
    fn into_inner(self) -> c_int { self.0.into_raw() }
}
