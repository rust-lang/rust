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
use net::{SocketAddr, Shutdown};
use str;
use sync::atomic::{AtomicBool, Ordering};
use sys::fd::FileDesc;
use sys_common::{AsInner, FromInner, IntoInner};
use sys_common::net::{getsockopt, setsockopt};
use time::Duration;

pub use sys::{cvt, cvt_r};
pub use libc as netc;

pub type wrlen_t = size_t;

pub struct Socket(FileDesc);

pub fn init() {}

pub fn cvt_gai(err: c_int) -> io::Result<()> {
    if err == 0 { return Ok(()) }

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
        // We want to atomically duplicate this file descriptor and set the
        // CLOEXEC flag, and currently that's done via F_DUPFD_CLOEXEC. This
        // flag, however, isn't supported on older Linux kernels (earlier than
        // 2.6.24).
        //
        // To detect this and ensure that CLOEXEC is still set, we
        // follow a strategy similar to musl [1] where if passing
        // F_DUPFD_CLOEXEC causes `fcntl` to return EINVAL it means it's not
        // supported (the third parameter, 0, is always valid), so we stop
        // trying that. We also *still* call the `set_cloexec` method as
        // apparently some kernel at some point stopped setting CLOEXEC even
        // though it reported doing so on F_DUPFD_CLOEXEC.
        //
        // Also note that Android doesn't have F_DUPFD_CLOEXEC, but get it to
        // resolve so we at least compile this.
        //
        // [1]: http://comments.gmane.org/gmane.linux.lib.musl.general/2963
        #[cfg(target_os = "android")]
        use libc::F_DUPFD as F_DUPFD_CLOEXEC;
        #[cfg(not(target_os = "android"))]
        use libc::F_DUPFD_CLOEXEC;

        let make_socket = |fd| {
            let fd = FileDesc::new(fd);
            fd.set_cloexec();
            Socket(fd)
        };
        static TRY_CLOEXEC: AtomicBool = AtomicBool::new(true);
        let fd = self.0.raw();
        if !cfg!(target_os = "android") && TRY_CLOEXEC.load(Ordering::Relaxed) {
            match cvt(unsafe { libc::fcntl(fd, F_DUPFD_CLOEXEC, 0) }) {
                Err(ref e) if e.raw_os_error() == Some(libc::EINVAL) => {
                    TRY_CLOEXEC.store(false, Ordering::Relaxed);
                }
                res => return res.map(make_socket),
            }
        }
        cvt(unsafe { libc::fcntl(fd, libc::F_DUPFD, 0) }).map(make_socket)
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

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        let how = match how {
            Shutdown::Write => libc::SHUT_WR,
            Shutdown::Read => libc::SHUT_RD,
            Shutdown::Both => libc::SHUT_RDWR,
        };
        try!(cvt(unsafe { libc::shutdown(self.0.raw(), how) }));
        Ok(())
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
