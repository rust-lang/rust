// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io;
use libc::{c_int, c_void};
use mem;
use net::{SocketAddr, Shutdown};
use num::One;
use ops::Neg;
use ptr;
use sync::Once;
use sys::c;
use sys;
use sys_common::{self, AsInner, FromInner, IntoInner};
use sys_common::net;
use time::Duration;

pub type wrlen_t = i32;

pub mod netc {
    pub use sys::c::*;
    pub use sys::c::SOCKADDR as sockaddr;
    pub use sys::c::SOCKADDR_STORAGE_LH as sockaddr_storage;
    pub use sys::c::ADDRINFOA as addrinfo;
    pub use sys::c::ADDRESS_FAMILY as sa_family_t;
}

pub struct Socket(c::SOCKET);

/// Checks whether the Windows socket interface has been started already, and
/// if not, starts it.
pub fn init() {
    static START: Once = Once::new();

    START.call_once(|| unsafe {
        let mut data: c::WSADATA = mem::zeroed();
        let ret = c::WSAStartup(0x202, // version 2.2
                                &mut data);
        assert_eq!(ret, 0);

        let _ = sys_common::at_exit(|| { c::WSACleanup(); });
    });
}

/// Returns the last error from the Windows socket interface.
fn last_error() -> io::Error {
    io::Error::from_raw_os_error(unsafe { c::WSAGetLastError() })
}

/// Checks if the signed integer is the Windows constant `SOCKET_ERROR` (-1)
/// and if so, returns the last error from the Windows socket interface. . This
/// function must be called before another call to the socket API is made.
pub fn cvt<T: One + Neg<Output=T> + PartialEq>(t: T) -> io::Result<T> {
    let one: T = T::one();
    if t == -one {
        Err(last_error())
    } else {
        Ok(t)
    }
}

/// Provides the functionality of `cvt` for the return values of `getaddrinfo`
/// and similar, meaning that they return an error if the return value is 0.
pub fn cvt_gai(err: c_int) -> io::Result<()> {
    if err == 0 { return Ok(()) }
    cvt(err).map(|_| ())
}

/// Provides the functionality of `cvt` for a closure.
pub fn cvt_r<T, F>(mut f: F) -> io::Result<T>
    where F: FnMut() -> T, T: One + Neg<Output=T> + PartialEq
{
    cvt(f())
}

impl Socket {
    pub fn new(addr: &SocketAddr, ty: c_int) -> io::Result<Socket> {
        let fam = match *addr {
            SocketAddr::V4(..) => c::AF_INET,
            SocketAddr::V6(..) => c::AF_INET6,
        };
        let socket = try!(unsafe {
            match c::WSASocketW(fam, ty, 0, ptr::null_mut(), 0,
                                c::WSA_FLAG_OVERLAPPED) {
                c::INVALID_SOCKET => Err(last_error()),
                n => Ok(Socket(n)),
            }
        });
        try!(socket.set_no_inherit());
        Ok(socket)
    }

    pub fn accept(&self, storage: *mut c::SOCKADDR,
                  len: *mut c_int) -> io::Result<Socket> {
        let socket = try!(unsafe {
            match c::accept(self.0, storage, len) {
                c::INVALID_SOCKET => Err(last_error()),
                n => Ok(Socket(n)),
            }
        });
        try!(socket.set_no_inherit());
        Ok(socket)
    }

    pub fn duplicate(&self) -> io::Result<Socket> {
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
                c::INVALID_SOCKET => Err(last_error()),
                n => Ok(Socket(n)),
            }
        });
        try!(socket.set_no_inherit());
        Ok(socket)
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        // On unix when a socket is shut down all further reads return 0, so we
        // do the same on windows to map a shut down socket to returning EOF.
        unsafe {
            match c::recv(self.0, buf.as_mut_ptr() as *mut c_void,
                             buf.len() as i32, 0) {
                -1 if c::WSAGetLastError() == c::WSAESHUTDOWN => Ok(0),
                -1 => Err(last_error()),
                n => Ok(n as usize)
            }
        }
    }

    pub fn set_timeout(&self, dur: Option<Duration>,
                       kind: c_int) -> io::Result<()> {
        let timeout = match dur {
            Some(dur) => {
                let timeout = sys::dur2timeout(dur);
                if timeout == 0 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput,
                                              "cannot set a 0 duration timeout"));
                }
                timeout
            }
            None => 0
        };
        net::setsockopt(self, c::SOL_SOCKET, kind, timeout)
    }

    pub fn timeout(&self, kind: c_int) -> io::Result<Option<Duration>> {
        let raw: c::DWORD = try!(net::getsockopt(self, c::SOL_SOCKET, kind));
        if raw == 0 {
            Ok(None)
        } else {
            let secs = raw / 1000;
            let nsec = (raw % 1000) * 1000000;
            Ok(Some(Duration::new(secs as u64, nsec as u32)))
        }
    }

    fn set_no_inherit(&self) -> io::Result<()> {
        sys::cvt(unsafe {
            c::SetHandleInformation(self.0 as c::HANDLE,
                                    c::HANDLE_FLAG_INHERIT, 0)
        }).map(|_| ())
    }

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        let how = match how {
            Shutdown::Write => c::SD_SEND,
            Shutdown::Read => c::SD_RECEIVE,
            Shutdown::Both => c::SD_BOTH,
        };
        try!(cvt(unsafe { c::shutdown(self.0, how) }));
        Ok(())
    }
}

impl Drop for Socket {
    fn drop(&mut self) {
        let _ = unsafe { c::closesocket(self.0) };
    }
}

impl AsInner<c::SOCKET> for Socket {
    fn as_inner(&self) -> &c::SOCKET { &self.0 }
}

impl FromInner<c::SOCKET> for Socket {
    fn from_inner(sock: c::SOCKET) -> Socket { Socket(sock) }
}

impl IntoInner<c::SOCKET> for Socket {
    fn into_inner(self) -> c::SOCKET {
        let ret = self.0;
        mem::forget(self);
        ret
    }
}
