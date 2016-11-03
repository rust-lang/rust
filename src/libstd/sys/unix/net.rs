// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ffi::CStr;
use io;
use libc::{self, c_int, size_t, sockaddr, socklen_t, EAI_SYSTEM};
use net::{SocketAddr, Shutdown};
use str;
use sys::fd::FileDesc;
use sys_common::{AsInner, FromInner, IntoInner};
use sys_common::net::{getsockopt, setsockopt};
use time::Duration;

pub use sys::{cvt, cvt_r};
pub extern crate libc as netc;

pub type wrlen_t = size_t;

// See below for the usage of SOCK_CLOEXEC, but this constant is only defined on
// Linux currently (e.g. support doesn't exist on other platforms). In order to
// get name resolution to work and things to compile we just define a dummy
// SOCK_CLOEXEC here for other platforms. Note that the dummy constant isn't
// actually ever used (the blocks below are wrapped in `if cfg!` as well.
#[cfg(target_os = "linux")]
use libc::SOCK_CLOEXEC;
#[cfg(not(target_os = "linux"))]
const SOCK_CLOEXEC: c_int = 0;

// Another conditional contant for name resolution: Macos et iOS use
// SO_NOSIGPIPE as a setsockopt flag to disable SIGPIPE emission on socket.
// Other platforms do otherwise.
#[cfg(target_vendor = "apple")]
use libc::SO_NOSIGPIPE;
#[cfg(not(target_vendor = "apple"))]
const SO_NOSIGPIPE: c_int = 0;

pub struct Socket(FileDesc);

pub fn init() {}

pub fn cvt_gai(err: c_int) -> io::Result<()> {
    if err == 0 {
        return Ok(())
    }
    if err == EAI_SYSTEM {
        return Err(io::Error::last_os_error())
    }

    let detail = unsafe {
        str::from_utf8(CStr::from_ptr(libc::gai_strerror(err)).to_bytes()).unwrap()
            .to_owned()
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
        Socket::new_raw(fam, ty)
    }

    pub fn new_raw(fam: c_int, ty: c_int) -> io::Result<Socket> {
        unsafe {
            // On linux we first attempt to pass the SOCK_CLOEXEC flag to
            // atomically create the socket and set it as CLOEXEC. Support for
            // this option, however, was added in 2.6.27, and we still support
            // 2.6.18 as a kernel, so if the returned error is EINVAL we
            // fallthrough to the fallback.
            if cfg!(target_os = "linux") {
                match cvt(libc::socket(fam, ty | SOCK_CLOEXEC, 0)) {
                    Ok(fd) => return Ok(Socket(FileDesc::new(fd))),
                    Err(ref e) if e.raw_os_error() == Some(libc::EINVAL) => {}
                    Err(e) => return Err(e),
                }
            }

            let fd = cvt(libc::socket(fam, ty, 0))?;
            let fd = FileDesc::new(fd);
            fd.set_cloexec()?;
            let socket = Socket(fd);
            if cfg!(target_vendor = "apple") {
                setsockopt(&socket, libc::SOL_SOCKET, SO_NOSIGPIPE, 1)?;
            }
            Ok(socket)
        }
    }

    pub fn new_pair(fam: c_int, ty: c_int) -> io::Result<(Socket, Socket)> {
        unsafe {
            let mut fds = [0, 0];

            // Like above, see if we can set cloexec atomically
            if cfg!(target_os = "linux") {
                match cvt(libc::socketpair(fam, ty | SOCK_CLOEXEC, 0, fds.as_mut_ptr())) {
                    Ok(_) => {
                        return Ok((Socket(FileDesc::new(fds[0])), Socket(FileDesc::new(fds[1]))));
                    }
                    Err(ref e) if e.raw_os_error() == Some(libc::EINVAL) => {},
                    Err(e) => return Err(e),
                }
            }

            cvt(libc::socketpair(fam, ty, 0, fds.as_mut_ptr()))?;
            let a = FileDesc::new(fds[0]);
            let b = FileDesc::new(fds[1]);
            a.set_cloexec()?;
            b.set_cloexec()?;
            Ok((Socket(a), Socket(b)))
        }
    }

    pub fn accept(&self, storage: *mut sockaddr, len: *mut socklen_t)
                  -> io::Result<Socket> {
        // Unfortunately the only known way right now to accept a socket and
        // atomically set the CLOEXEC flag is to use the `accept4` syscall on
        // Linux. This was added in 2.6.28, however, and because we support
        // 2.6.18 we must detect this support dynamically.
        if cfg!(target_os = "linux") {
            weak! {
                fn accept4(c_int, *mut sockaddr, *mut socklen_t, c_int) -> c_int
            }
            if let Some(accept) = accept4.get() {
                let res = cvt_r(|| unsafe {
                    accept(self.0.raw(), storage, len, SOCK_CLOEXEC)
                });
                match res {
                    Ok(fd) => return Ok(Socket(FileDesc::new(fd))),
                    Err(ref e) if e.raw_os_error() == Some(libc::ENOSYS) => {}
                    Err(e) => return Err(e),
                }
            }
        }

        let fd = cvt_r(|| unsafe {
            libc::accept(self.0.raw(), storage, len)
        })?;
        let fd = FileDesc::new(fd);
        fd.set_cloexec()?;
        Ok(Socket(fd))
    }

    pub fn duplicate(&self) -> io::Result<Socket> {
        self.0.duplicate().map(Socket)
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
        let raw: libc::timeval = getsockopt(self, libc::SOL_SOCKET, kind)?;
        if raw.tv_sec == 0 && raw.tv_usec == 0 {
            Ok(None)
        } else {
            let sec = raw.tv_sec as u64;
            let nsec = (raw.tv_usec as u32) * 1000;
            Ok(Some(Duration::new(sec, nsec)))
        }
    }

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        let how = match how {
            Shutdown::Write => libc::SHUT_WR,
            Shutdown::Read => libc::SHUT_RD,
            Shutdown::Both => libc::SHUT_RDWR,
        };
        cvt(unsafe { libc::shutdown(self.0.raw(), how) })?;
        Ok(())
    }

    pub fn set_nodelay(&self, nodelay: bool) -> io::Result<()> {
        setsockopt(self, libc::IPPROTO_TCP, libc::TCP_NODELAY, nodelay as c_int)
    }

    pub fn nodelay(&self) -> io::Result<bool> {
        let raw: c_int = getsockopt(self, libc::IPPROTO_TCP, libc::TCP_NODELAY)?;
        Ok(raw != 0)
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        let mut nonblocking = nonblocking as libc::c_int;
        cvt(unsafe { libc::ioctl(*self.as_inner(), libc::FIONBIO, &mut nonblocking) }).map(|_| ())
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        let raw: c_int = getsockopt(self, libc::SOL_SOCKET, libc::SO_ERROR)?;
        if raw == 0 {
            Ok(None)
        } else {
            Ok(Some(io::Error::from_raw_os_error(raw as i32)))
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
