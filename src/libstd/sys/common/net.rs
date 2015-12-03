// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;

use ffi::{CStr, CString};
use fmt;
use io::{self, Error, ErrorKind};
use libc::{c_int, c_char, c_void};
use mem;
#[allow(deprecated)]
use net::{SocketAddr, Shutdown, IpAddr};
use ptr;
use str::from_utf8;
use sys::net::{cvt, cvt_r, cvt_gai, Socket, init, wrlen_t};
use sys::net::netc as c;
use sys_common::{AsInner, FromInner, IntoInner};
use time::Duration;

////////////////////////////////////////////////////////////////////////////////
// sockaddr and misc bindings
////////////////////////////////////////////////////////////////////////////////

pub fn setsockopt<T>(sock: &Socket, opt: c_int, val: c_int,
                     payload: T) -> io::Result<()> {
    unsafe {
        let payload = &payload as *const T as *const c_void;
        try!(cvt(c::setsockopt(*sock.as_inner(), opt, val, payload,
                               mem::size_of::<T>() as c::socklen_t)));
        Ok(())
    }
}

pub fn getsockopt<T: Copy>(sock: &Socket, opt: c_int,
                       val: c_int) -> io::Result<T> {
    unsafe {
        let mut slot: T = mem::zeroed();
        let mut len = mem::size_of::<T>() as c::socklen_t;
        try!(cvt(c::getsockopt(*sock.as_inner(), opt, val,
                               &mut slot as *mut _ as *mut _,
                               &mut len)));
        assert_eq!(len as usize, mem::size_of::<T>());
        Ok(slot)
    }
}

fn sockname<F>(f: F) -> io::Result<SocketAddr>
    where F: FnOnce(*mut c::sockaddr, *mut c::socklen_t) -> c_int
{
    unsafe {
        let mut storage: c::sockaddr_storage = mem::zeroed();
        let mut len = mem::size_of_val(&storage) as c::socklen_t;
        try!(cvt(f(&mut storage as *mut _ as *mut _, &mut len)));
        sockaddr_to_addr(&storage, len as usize)
    }
}

fn sockaddr_to_addr(storage: &c::sockaddr_storage,
                    len: usize) -> io::Result<SocketAddr> {
    match storage.ss_family as c_int {
        c::AF_INET => {
            assert!(len as usize >= mem::size_of::<c::sockaddr_in>());
            Ok(SocketAddr::V4(FromInner::from_inner(unsafe {
                *(storage as *const _ as *const c::sockaddr_in)
            })))
        }
        c::AF_INET6 => {
            assert!(len as usize >= mem::size_of::<c::sockaddr_in6>());
            Ok(SocketAddr::V6(FromInner::from_inner(unsafe {
                *(storage as *const _ as *const c::sockaddr_in6)
            })))
        }
        _ => {
            Err(Error::new(ErrorKind::InvalidInput, "invalid argument"))
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// get_host_addresses
////////////////////////////////////////////////////////////////////////////////

pub struct LookupHost {
    original: *mut c::addrinfo,
    cur: *mut c::addrinfo,
}

impl Iterator for LookupHost {
    type Item = io::Result<SocketAddr>;
    fn next(&mut self) -> Option<io::Result<SocketAddr>> {
        unsafe {
            if self.cur.is_null() { return None }
            let ret = sockaddr_to_addr(mem::transmute((*self.cur).ai_addr),
                                       (*self.cur).ai_addrlen as usize);
            self.cur = (*self.cur).ai_next as *mut c::addrinfo;
            Some(ret)
        }
    }
}

unsafe impl Sync for LookupHost {}
unsafe impl Send for LookupHost {}

impl Drop for LookupHost {
    fn drop(&mut self) {
        unsafe { c::freeaddrinfo(self.original) }
    }
}

pub fn lookup_host(host: &str) -> io::Result<LookupHost> {
    init();

    let c_host = try!(CString::new(host));
    let mut res = ptr::null_mut();
    unsafe {
        try!(cvt_gai(c::getaddrinfo(c_host.as_ptr(), ptr::null(), ptr::null(),
                                   &mut res)));
        Ok(LookupHost { original: res, cur: res })
    }
}

////////////////////////////////////////////////////////////////////////////////
// lookup_addr
////////////////////////////////////////////////////////////////////////////////

#[allow(deprecated)]
pub fn lookup_addr(addr: &IpAddr) -> io::Result<String> {
    init();

    let saddr = SocketAddr::new(*addr, 0);
    let (inner, len) = saddr.into_inner();
    let mut hostbuf = [0 as c_char; c::NI_MAXHOST as usize];

    let data = unsafe {
        try!(cvt_gai(c::getnameinfo(inner, len,
                                    hostbuf.as_mut_ptr(),
                                    c::NI_MAXHOST,
                                    ptr::null_mut(), 0, 0)));

        CStr::from_ptr(hostbuf.as_ptr())
    };

    match from_utf8(data.to_bytes()) {
        Ok(name) => Ok(name.to_owned()),
        Err(_) => Err(io::Error::new(io::ErrorKind::Other,
                                     "failed to lookup address information"))
    }
}

////////////////////////////////////////////////////////////////////////////////
// TCP streams
////////////////////////////////////////////////////////////////////////////////

pub struct TcpStream {
    inner: Socket,
}

impl TcpStream {
    pub fn connect(addr: &SocketAddr) -> io::Result<TcpStream> {
        init();

        let sock = try!(Socket::new(addr, c::SOCK_STREAM));

        let (addrp, len) = addr.into_inner();
        try!(cvt_r(|| unsafe { c::connect(*sock.as_inner(), addrp, len) }));
        Ok(TcpStream { inner: sock })
    }

    pub fn socket(&self) -> &Socket { &self.inner }

    pub fn into_socket(self) -> Socket { self.inner }

    pub fn set_read_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        self.inner.set_timeout(dur, c::SO_RCVTIMEO)
    }

    pub fn set_write_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        self.inner.set_timeout(dur, c::SO_SNDTIMEO)
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        self.inner.timeout(c::SO_RCVTIMEO)
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        self.inner.timeout(c::SO_SNDTIMEO)
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let ret = try!(cvt(unsafe {
            c::send(*self.inner.as_inner(),
                    buf.as_ptr() as *const c_void,
                    buf.len() as wrlen_t,
                    0)
        }));
        Ok(ret as usize)
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        sockname(|buf, len| unsafe {
            c::getpeername(*self.inner.as_inner(), buf, len)
        })
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        sockname(|buf, len| unsafe {
            c::getsockname(*self.inner.as_inner(), buf, len)
        })
    }

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        self.inner.shutdown(how)
    }

    pub fn duplicate(&self) -> io::Result<TcpStream> {
        self.inner.duplicate().map(|s| TcpStream { inner: s })
    }
}

impl FromInner<Socket> for TcpStream {
    fn from_inner(socket: Socket) -> TcpStream {
        TcpStream { inner: socket }
    }
}

impl fmt::Debug for TcpStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut res = f.debug_struct("TcpStream");

        if let Ok(addr) = self.socket_addr() {
            res.field("addr", &addr);
        }

        if let Ok(peer) = self.peer_addr() {
            res.field("peer", &peer);
        }

        let name = if cfg!(windows) {"socket"} else {"fd"};
        res.field(name, &self.inner.as_inner())
            .finish()
    }
}

////////////////////////////////////////////////////////////////////////////////
// TCP listeners
////////////////////////////////////////////////////////////////////////////////

pub struct TcpListener {
    inner: Socket,
}

impl TcpListener {
    pub fn bind(addr: &SocketAddr) -> io::Result<TcpListener> {
        init();

        let sock = try!(Socket::new(addr, c::SOCK_STREAM));

        // On platforms with Berkeley-derived sockets, this allows
        // to quickly rebind a socket, without needing to wait for
        // the OS to clean up the previous one.
        if !cfg!(windows) {
            try!(setsockopt(&sock, c::SOL_SOCKET, c::SO_REUSEADDR,
                            1 as c_int));
        }

        // Bind our new socket
        let (addrp, len) = addr.into_inner();
        try!(cvt(unsafe { c::bind(*sock.as_inner(), addrp, len) }));

        // Start listening
        try!(cvt(unsafe { c::listen(*sock.as_inner(), 128) }));
        Ok(TcpListener { inner: sock })
    }

    pub fn socket(&self) -> &Socket { &self.inner }

    pub fn into_socket(self) -> Socket { self.inner }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        sockname(|buf, len| unsafe {
            c::getsockname(*self.inner.as_inner(), buf, len)
        })
    }

    pub fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        let mut storage: c::sockaddr_storage = unsafe { mem::zeroed() };
        let mut len = mem::size_of_val(&storage) as c::socklen_t;
        let sock = try!(self.inner.accept(&mut storage as *mut _ as *mut _,
                                          &mut len));
        let addr = try!(sockaddr_to_addr(&storage, len as usize));
        Ok((TcpStream { inner: sock, }, addr))
    }

    pub fn duplicate(&self) -> io::Result<TcpListener> {
        self.inner.duplicate().map(|s| TcpListener { inner: s })
    }
}

impl FromInner<Socket> for TcpListener {
    fn from_inner(socket: Socket) -> TcpListener {
        TcpListener { inner: socket }
    }
}

impl fmt::Debug for TcpListener {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut res = f.debug_struct("TcpListener");

        if let Ok(addr) = self.socket_addr() {
            res.field("addr", &addr);
        }

        let name = if cfg!(windows) {"socket"} else {"fd"};
        res.field(name, &self.inner.as_inner())
            .finish()
    }
}

////////////////////////////////////////////////////////////////////////////////
// UDP
////////////////////////////////////////////////////////////////////////////////

pub struct UdpSocket {
    inner: Socket,
}

impl UdpSocket {
    pub fn bind(addr: &SocketAddr) -> io::Result<UdpSocket> {
        init();

        let sock = try!(Socket::new(addr, c::SOCK_DGRAM));
        let (addrp, len) = addr.into_inner();
        try!(cvt(unsafe { c::bind(*sock.as_inner(), addrp, len) }));
        Ok(UdpSocket { inner: sock })
    }

    pub fn socket(&self) -> &Socket { &self.inner }

    pub fn into_socket(self) -> Socket { self.inner }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        sockname(|buf, len| unsafe {
            c::getsockname(*self.inner.as_inner(), buf, len)
        })
    }

    pub fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        let mut storage: c::sockaddr_storage = unsafe { mem::zeroed() };
        let mut addrlen = mem::size_of_val(&storage) as c::socklen_t;

        let n = try!(cvt(unsafe {
            c::recvfrom(*self.inner.as_inner(),
                        buf.as_mut_ptr() as *mut c_void,
                        buf.len() as wrlen_t, 0,
                        &mut storage as *mut _ as *mut _, &mut addrlen)
        }));
        Ok((n as usize, try!(sockaddr_to_addr(&storage, addrlen as usize))))
    }

    pub fn send_to(&self, buf: &[u8], dst: &SocketAddr) -> io::Result<usize> {
        let (dstp, dstlen) = dst.into_inner();
        let ret = try!(cvt(unsafe {
            c::sendto(*self.inner.as_inner(),
                      buf.as_ptr() as *const c_void, buf.len() as wrlen_t,
                      0, dstp, dstlen)
        }));
        Ok(ret as usize)
    }

    pub fn duplicate(&self) -> io::Result<UdpSocket> {
        self.inner.duplicate().map(|s| UdpSocket { inner: s })
    }

    pub fn set_read_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        self.inner.set_timeout(dur, c::SO_RCVTIMEO)
    }

    pub fn set_write_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        self.inner.set_timeout(dur, c::SO_SNDTIMEO)
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        self.inner.timeout(c::SO_RCVTIMEO)
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        self.inner.timeout(c::SO_SNDTIMEO)
    }
}

impl FromInner<Socket> for UdpSocket {
    fn from_inner(socket: Socket) -> UdpSocket {
        UdpSocket { inner: socket }
    }
}

impl fmt::Debug for UdpSocket {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut res = f.debug_struct("UdpSocket");

        if let Ok(addr) = self.socket_addr() {
            res.field("addr", &addr);
        }

        let name = if cfg!(windows) {"socket"} else {"fd"};
        res.field(name, &self.inner.as_inner())
            .finish()
    }
}
