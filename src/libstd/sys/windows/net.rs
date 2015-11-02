// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc::consts::os::extra::INVALID_SOCKET;
use libc::{self, c_int, c_void};
use mem;
use num::One;
use ops::Neg;
use ptr;
use sync::Once;
use sys::common::net_bsd::{setsockopt, getsockopt};
use sys::error::{Result, Error};
use sys::inner::{AsInner, FromInner, IntoInner};
use sys::windows::c;
use time::Duration;

pub use libc::WSAEINVAL as EINVAL;
pub use sys::common::net::{IpAddr, SocketAddr};
pub use sys::common::net_bsd::{
    TcpListener, TcpStream, UdpSocket,
    IpAddrV4, IpAddrV6,
    SocketAddrV4, SocketAddrV6,
    LookupHost, LookupAddr,
    connect_tcp, bind_tcp, bind_udp,
    lookup_host, lookup_addr,
};

pub type wrlen_t = i32;

#[derive(Debug)]
pub struct Socket(libc::SOCKET);

/// Checks whether the Windows socket interface has been started already, and
/// if not, starts it.
pub fn init() {
    static START: Once = Once::new();

    START.call_once(|| unsafe {
        use at_exit::at_exit;

        let mut data: c::WSADATA = mem::zeroed();
        let ret = c::WSAStartup(0x202, // version 2.2
                                &mut data);
        assert_eq!(ret, 0);

        let _ = at_exit(|| { c::WSACleanup(); });
    });
}

/// Returns the last error from the Windows socket interface.
fn last_error() -> Error {
    Error::from_code(unsafe { c::WSAGetLastError() })
}

/// Checks if the signed integer is the Windows constant `SOCKET_ERROR` (-1)
/// and if so, returns the last error from the Windows socket interface. . This
/// function must be called before another call to the socket API is made.
pub fn cvt<T: One + Neg<Output=T> + PartialEq>(t: T) -> Result<T> {
    let one: T = T::one();
    if t == -one {
        Err(last_error())
    } else {
        Ok(t)
    }
}

/// Provides the functionality of `cvt` for the return values of `getaddrinfo`
/// and similar, meaning that they return an error if the return value is 0.
pub fn cvt_gai(err: c_int) -> Result<()> {
    if err == 0 { return Ok(()) }
    cvt(err).map(|_| ())
}

/// Provides the functionality of `cvt` for a closure.
pub fn cvt_r<T, F>(mut f: F) -> Result<T>
    where F: FnMut() -> T, T: One + Neg<Output=T> + PartialEq
{
    cvt(f())
}

impl Socket {
    pub fn new(addr: &SocketAddr, ty: c_int) -> Result<Socket> {
        let fam = match *addr {
            SocketAddr::V4(..) => libc::AF_INET,
            SocketAddr::V6(..) => libc::AF_INET6,
        };
        let socket = try!(unsafe {
            match c::WSASocketW(fam, ty, 0, ptr::null_mut(), 0,
                                c::WSA_FLAG_OVERLAPPED) {
                INVALID_SOCKET => Err(last_error()),
                n => Ok(Socket(n)),
            }
        });
        try!(socket.set_no_inherit());
        Ok(socket)
    }

    pub fn accept(&self, storage: *mut libc::sockaddr,
                  len: *mut libc::socklen_t) -> Result<Socket> {
        let socket = try!(unsafe {
            match libc::accept(self.0, storage, len) {
                INVALID_SOCKET => Err(last_error()),
                n => Ok(Socket(n)),
            }
        });
        try!(socket.set_no_inherit());
        Ok(socket)
    }

    pub fn duplicate(&self) -> Result<Socket> {
        let socket = try!(unsafe {
            let mut info: c::WSAPROTOCOL_INFO = mem::zeroed();
            try!(cvt(c::WSADuplicateSocketW(self.0,
                                            c::GetCurrentProcessId(),
                                            &mut info)));
            match c::WSASocketW(info.iAddressFamily,
                                info.iSocketType,
                                info.iProtocol,
                                &mut info, 0,
                                c::WSA_FLAG_OVERLAPPED) {
                INVALID_SOCKET => Err(last_error()),
                n => Ok(Socket(n)),
            }
        });
        try!(socket.set_no_inherit());
        Ok(socket)
    }

    pub fn read(&self, buf: &mut [u8]) -> Result<usize> {
        // On unix when a socket is shut down all further reads return 0, so we
        // do the same on windows to map a shut down socket to returning EOF.
        unsafe {
            match libc::recv(self.0, buf.as_mut_ptr() as *mut c_void,
                             buf.len() as i32, 0) {
                -1 if c::WSAGetLastError() == c::WSAESHUTDOWN => Ok(0),
                -1 => Err(last_error()),
                n => Ok(n as usize)
            }
        }
    }

    pub fn write(&self, buf: &[u8]) -> Result<usize> {
        cvt(unsafe {
            libc::send(self.0,
                       buf.as_ptr() as *const c_void,
                       buf.len() as wrlen_t,
                       0)
        }).map(|r| r as usize)
    }

    pub fn set_timeout(&self, dur: Option<Duration>, kind: libc::c_int) -> Result<()> {
        let timeout = match dur {
            Some(dur) => {
                let timeout = c::dur2timeout(dur);
                if timeout == 0 {
                    return Err(Error::from_code(EINVAL));
                }
                timeout
            }
            None => 0
        };
        setsockopt(self, libc::SOL_SOCKET, kind, timeout)
    }

    pub fn timeout(&self, kind: libc::c_int) -> Result<Option<Duration>> {
        let raw: libc::DWORD = try!(getsockopt(self, libc::SOL_SOCKET, kind));
        if raw == 0 {
            Ok(None)
        } else {
            let secs = raw / 1000;
            let nsec = (raw % 1000) * 1000000;
            Ok(Some(Duration::new(secs as u64, nsec as u32)))
        }
    }

    fn set_no_inherit(&self) -> Result<()> {
        c::cvt(unsafe {
            c::SetHandleInformation(self.0 as libc::HANDLE,
                                    c::HANDLE_FLAG_INHERIT, 0)
        }).map(|_| ())
    }
}

impl Drop for Socket {
    fn drop(&mut self) {
        let _ = unsafe { libc::closesocket(self.0) };
    }
}

impl AsInner<libc::SOCKET> for Socket {
    fn as_inner(&self) -> &libc::SOCKET { &self.0 }
}

impl FromInner<libc::SOCKET> for Socket {
    fn from_inner(sock: libc::SOCKET) -> Socket { Socket(sock) }
}

impl IntoInner<libc::SOCKET> for Socket {
    fn into_inner(self) -> libc::SOCKET {
        let ret = self.0;
        mem::forget(self);
        ret
    }
}
