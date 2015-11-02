// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io::prelude::*;
use sys::inner::*;

use fmt;
use libc::{self, c_int, size_t};
use sync::atomic::{AtomicBool, Ordering};
use sys::error::{Error, Result};
use sys::unix::fd::FileDesc;
use sys::common::net_bsd::{getsockopt, setsockopt};
use sys::net as sys;
use time::Duration;

pub use libc::EINVAL;
pub use sys::unix::{cvt, cvt_r};
pub use sys::common::net::{IpAddr, SocketAddr};
pub use sys::common::net_bsd::{
    TcpListener, TcpStream, UdpSocket,
    IpAddrV4, IpAddrV6,
    SocketAddrV4, SocketAddrV6,
    LookupHost, LookupAddr,
    connect_tcp, bind_tcp, bind_udp,
    lookup_host, lookup_addr,
};

pub type wrlen_t = size_t;

pub struct Socket(FileDesc);

pub fn init() {}

pub fn cvt_gai(err: c_int) -> Result<()> {
    if err == 0 { return Ok(()) }

    Err(Error::from_code(err))
}

impl Socket {
    pub fn new(addr: &sys::SocketAddr, ty: c_int) -> Result<Socket> {
        let fam = match *addr {
            sys::SocketAddr::V4(..) => libc::AF_INET,
            sys::SocketAddr::V6(..) => libc::AF_INET6,
        };
        unsafe {
            let fd = try!(cvt(libc::socket(fam, ty, 0)));
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
        if !EMULATE_F_DUPFD_CLOEXEC.load(Ordering::Relaxed) {
            match cvt(unsafe { fcntl(*self.0.as_inner(), libc::F_DUPFD_CLOEXEC, 0) }) {
                // `EINVAL` can only be returned on two occasions: Invalid
                // command (second parameter) or invalid third parameter. 0 is
                // always a valid third parameter, so it must be the second
                // parameter.
                //
                // Store the result in a global variable so we don't try each
                // syscall twice.
                Err(ref e) if e.code() == libc::EINVAL =>
                    EMULATE_F_DUPFD_CLOEXEC.store(true, Ordering::Relaxed),
                r => return r.map(make_socket),
            }
        }
        cvt(unsafe { fcntl(*self.0.as_inner(), libc::F_DUPFD, 0) }).map(make_socket)
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

    pub fn read(&self, buf: &mut [u8]) -> Result<usize> {
        self.0.read(buf)
    }

    pub fn write(&self, buf: &[u8]) -> Result<usize> {
        self.0.write(buf)
    }
}

impl_inner!(Socket(FileDesc));
impl_inner!(Socket(FileDesc(c_int)));

impl fmt::Debug for Socket {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self.0.as_inner(), f)
    }
}
