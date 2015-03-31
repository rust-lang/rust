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

use io;
use libc::consts::os::extra::INVALID_SOCKET;
use libc::{self, c_int, c_void};
use mem;
use net::SocketAddr;
#[allow(deprecated)]
use num::{SignedInt, Int};
use rt;
use sync::{Once, ONCE_INIT};
use sys::c;
use sys_common::AsInner;

pub type wrlen_t = i32;

pub struct Socket(libc::SOCKET);

/// Checks whether the Windows socket interface has been started already, and
/// if not, starts it.
pub fn init() {
    static START: Once = ONCE_INIT;

    START.call_once(|| unsafe {
        let mut data: c::WSADATA = mem::zeroed();
        let ret = c::WSAStartup(0x202, // version 2.2
                                &mut data);
        assert_eq!(ret, 0);

        let _ = rt::at_exit(|| { c::WSACleanup(); });
    });
}

/// Returns the last error from the Windows socket interface.
fn last_error() -> io::Error {
    io::Error::from_os_error(unsafe { c::WSAGetLastError() })
}

/// Checks if the signed integer is the Windows constant `SOCKET_ERROR` (-1)
/// and if so, returns the last error from the Windows socket interface. . This
/// function must be called before another call to the socket API is made.
///
/// FIXME: generics needed?
#[allow(deprecated)]
pub fn cvt<T: SignedInt>(t: T) -> io::Result<T> {
    let one: T = Int::one();
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
#[allow(deprecated)]
pub fn cvt_r<T: SignedInt, F>(mut f: F) -> io::Result<T> where F: FnMut() -> T {
    cvt(f())
}

impl Socket {
    pub fn new(addr: &SocketAddr, ty: c_int) -> io::Result<Socket> {
        let fam = match *addr {
            SocketAddr::V4(..) => libc::AF_INET,
            SocketAddr::V6(..) => libc::AF_INET6,
        };
        match unsafe { libc::socket(fam, ty, 0) } {
            INVALID_SOCKET => Err(last_error()),
            n => Ok(Socket(n)),
        }
    }

    pub fn accept(&self, storage: *mut libc::sockaddr,
                  len: *mut libc::socklen_t) -> io::Result<Socket> {
        match unsafe { libc::accept(self.0, storage, len) } {
            INVALID_SOCKET => Err(last_error()),
            n => Ok(Socket(n)),
        }
    }

    pub fn duplicate(&self) -> io::Result<Socket> {
        unsafe {
            let mut info: c::WSAPROTOCOL_INFO = mem::zeroed();
            try!(cvt(c::WSADuplicateSocketW(self.0,
                                            c::GetCurrentProcessId(),
                                            &mut info)));
            match c::WSASocketW(info.iAddressFamily,
                                info.iSocketType,
                                info.iProtocol,
                                &mut info, 0, 0) {
                INVALID_SOCKET => Err(last_error()),
                n => Ok(Socket(n)),
            }
        }
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
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
}

impl Drop for Socket {
    fn drop(&mut self) {
        unsafe { cvt(libc::closesocket(self.0)).unwrap(); }
    }
}

impl AsInner<libc::SOCKET> for Socket {
    fn as_inner(&self) -> &libc::SOCKET { &self.0 }
}
