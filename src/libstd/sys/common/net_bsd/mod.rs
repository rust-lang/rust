// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io::prelude::*;
use sys::error::{Error, Result};
use sys::inner::*;

use ffi::{CStr, CString};
use libc::{self, c_int, c_char, c_void, socklen_t};
use mem;
use ptr;
use io;
use sys::target::c;
use sys::target::net::{cvt, cvt_r, cvt_gai, Socket as SocketImp, init, wrlen_t};
use sys::common::net::{SocketAddr, IpAddr};
use sys::net as sys;
use net::Shutdown;
use time::Duration;

mod ip;
mod addr;

use self::addr::{new_sockaddr, sockaddr};

pub use self::ip::{IpAddrV4, IpAddrV6};
pub use self::addr::{SocketAddrV4, SocketAddrV6};

pub type TcpStream = Socket;
pub type TcpListener = Socket;
pub type UdpSocket = Socket;

pub fn setsockopt<T>(sock: &SocketImp, opt: c_int, val: c_int,
                     payload: T) -> Result<()> {
    unsafe {
        let payload = &payload as *const T as *const c_void;
        cvt(libc::setsockopt(*sock.as_inner(), opt, val, payload, mem::size_of::<T>() as socklen_t)).map(drop)
    }
}

pub fn getsockopt<T: Copy>(sock: &SocketImp, opt: c_int,
                       val: c_int) -> Result<T> {
    unsafe {
        let mut slot: T = mem::zeroed();
        let mut len = mem::size_of::<T>() as socklen_t;
        try!(cvt(c::getsockopt(*sock.as_inner(), opt, val, &mut slot as *mut _ as *mut _, &mut len)));
        if len as usize != mem::size_of::<T>() {
            Err(Error::from_code(sys::EINVAL))
        } else {
            Ok(slot)
        }
    }
}

fn sockname<F>(f: F) -> Result<SocketAddr>
    where F: FnOnce(*mut libc::sockaddr, *mut socklen_t) -> c_int
{
    unsafe {
        let mut storage: libc::sockaddr_storage = mem::zeroed();
        let mut len = mem::size_of_val(&storage) as socklen_t;
        try!(cvt(f(&mut storage as *mut _ as *mut _, &mut len)));
        sockaddr_to_addr(&storage, len as usize)
    }
}

fn sockaddr_to_addr(storage: &libc::sockaddr_storage,
                    len: usize) -> Result<SocketAddr> {
    match storage.ss_family as libc::c_int {
        libc::AF_INET => {
            assert!(len as usize >= mem::size_of::<libc::sockaddr_in>());
            Ok(SocketAddr::V4(FromInner::from_inner(unsafe {
                *(storage as *const _ as *const libc::sockaddr_in)
            })))
        }
        libc::AF_INET6 => {
            assert!(len as usize >= mem::size_of::<libc::sockaddr_in6>());
            Ok(SocketAddr::V6(FromInner::from_inner(unsafe {
                *(storage as *const _ as *const libc::sockaddr_in6)
            })))
        }
        _ => Err(Error::from_code(sys::EINVAL))
    }
}

trait NetInt {
    fn from_be(i: Self) -> Self;
    fn to_be(&self) -> Self;
}
macro_rules! doit {
    ($($t:ident)*) => ($(impl NetInt for $t {
        fn from_be(i: Self) -> Self { <$t>::from_be(i) }
        fn to_be(&self) -> Self { <$t>::to_be(*self) }
    })*)
}
doit! { i8 i16 i32 i64 isize u8 u16 u32 u64 usize }

fn hton<I: NetInt>(i: I) -> I { i.to_be() }
fn ntoh<I: NetInt>(i: I) -> I { I::from_be(i) }

extern "system" {
    fn getaddrinfo(node: *const c_char, service: *const c_char,
                   hints: *const libc::addrinfo,
                   res: *mut *mut libc::addrinfo) -> c_int;
    fn freeaddrinfo(res: *mut libc::addrinfo);
}

pub struct LookupHost {
    original: *mut libc::addrinfo,
    cur: *mut libc::addrinfo,
}

impl Iterator for LookupHost {
    type Item = Result<SocketAddr>;
    fn next(&mut self) -> Option<Result<SocketAddr>> {
        unsafe {
            if self.cur.is_null() { return None }
            let ret = sockaddr_to_addr(mem::transmute((*self.cur).ai_addr),
                                       (*self.cur).ai_addrlen as usize);
            self.cur = (*self.cur).ai_next as *mut libc::addrinfo;
            Some(ret)
        }
    }
}

unsafe impl Sync for LookupHost {}
unsafe impl Send for LookupHost {}

impl Drop for LookupHost {
    fn drop(&mut self) {
        unsafe { freeaddrinfo(self.original) }
    }
}

extern "system" {
    fn getnameinfo(sa: *const libc::sockaddr, salen: socklen_t,
                   host: *mut c_char, hostlen: libc::size_t,
                   serv: *mut c_char, servlen: libc::size_t,
                   flags: c_int) -> c_int;
}

const NI_MAXHOST: usize = 1025;

pub struct LookupAddr([c_char; NI_MAXHOST]);

impl LookupAddr {
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { CStr::from_ptr(self.0.as_ptr()).to_bytes() }
    }
}

pub struct Socket(SocketImp);
impl_inner!(Socket(SocketImp));
impl_inner!(for<T> Socket(SocketImp(T)));

impl Socket {
    pub fn set_read_timeout(&self, dur: Option<Duration>) -> Result<()> {
        self.0.set_timeout(dur, libc::SO_RCVTIMEO)
    }

    pub fn set_write_timeout(&self, dur: Option<Duration>) -> Result<()> {
        self.0.set_timeout(dur, libc::SO_SNDTIMEO)
    }

    pub fn read_timeout(&self) -> Result<Option<Duration>> {
        self.0.timeout(libc::SO_RCVTIMEO)
    }

    pub fn write_timeout(&self) -> Result<Option<Duration>> {
        self.0.timeout(libc::SO_SNDTIMEO)
    }

    pub fn duplicate(&self) -> Result<Self> {
        self.0.duplicate().map(Socket)
    }

    pub fn socket_addr(&self) -> Result<SocketAddr> {
        sockname(|buf, len| unsafe {
            libc::getsockname(*self.0.as_inner(), buf, len)
        })
    }

    pub fn read(&self, buf: &mut [u8]) -> Result<usize> {
        self.0.read(buf)
    }

    pub fn write(&self, buf: &[u8]) -> Result<usize> {
        self.0.write(buf)
    }

    pub fn flush(&self) -> io::Result<()> { Ok(()) }
}

impl Socket {
    pub fn peer_addr(&self) -> Result<SocketAddr> {
        sockname(|buf, len| unsafe {
            libc::getpeername(*self.0.as_inner(), buf, len)
        })
    }

    pub fn shutdown(&self, how: Shutdown) -> Result<()> {
        use libc::consts::os::bsd44::SHUT_RDWR;

        let how = match how {
            Shutdown::Write => libc::SHUT_WR,
            Shutdown::Read => libc::SHUT_RD,
            Shutdown::Both => SHUT_RDWR,
        };
        cvt(unsafe { libc::shutdown(*self.0.as_inner(), how) })
            .map_err(From::from).map(drop)
    }
}

impl Socket {
    pub fn accept(&self) -> Result<(Socket, SocketAddr)> {
        let mut storage: libc::sockaddr_storage = unsafe { mem::zeroed() };
        let mut len = mem::size_of_val(&storage) as socklen_t;
        let sock = try!(self.0.accept(&mut storage as *mut _ as *mut _,
                                          &mut len));
        let addr = try!(sockaddr_to_addr(&storage, len as usize));
        Ok((Socket(sock), addr))
    }
}

impl Socket {
    pub fn recv_from(&self, buf: &mut [u8]) -> Result<(usize, SocketAddr)> {
        let mut storage: libc::sockaddr_storage = unsafe { mem::zeroed() };
        let mut addrlen = mem::size_of_val(&storage) as socklen_t;

        let n = try!(cvt(unsafe { libc::recvfrom(*self.0.as_inner(),
            buf.as_mut_ptr() as *mut c_void, buf.len() as wrlen_t, 0, &mut storage as *mut _ as *mut _, &mut addrlen) }));
        Ok((n as usize, try!(sockaddr_to_addr(&storage, addrlen as usize))))
    }

    pub fn send_to(&self, buf: &[u8], dst: &SocketAddr) -> Result<usize> {
        let (dstp, dstlen) = sockaddr(dst);
        cvt(unsafe { libc::sendto(*self.0.as_inner(), buf.as_ptr() as *const c_void, buf.len() as wrlen_t, 0, dstp, dstlen) })
            .map_err(From::from).map(|v| v as usize)
    }
}

pub fn lookup_host(host: &str) -> Result<LookupHost> {
    init();

    let c_host = try!(CString::new(host).map_err(|_| Error::from_code(sys::EINVAL)));
    let mut res = ptr::null_mut();
    unsafe {
        try!(cvt_gai(getaddrinfo(c_host.as_ptr(), ptr::null(), ptr::null(),
                                 &mut res)));
        Ok(LookupHost { original: res, cur: res })
    }
}

pub fn lookup_addr(addr: &sys::IpAddr) -> Result<LookupAddr> {
    init();

    let saddr = new_sockaddr(addr.clone(), 0);
    let (inner, len) = sockaddr(&saddr);
    let mut hostbuf = [0 as c_char; NI_MAXHOST];

    unsafe {
        try!(cvt_gai(getnameinfo(inner, len,
                                 hostbuf.as_mut_ptr(), NI_MAXHOST as libc::size_t,
                                 ptr::null_mut(), 0, 0)));
    }

    Ok(LookupAddr(hostbuf))
}

pub fn connect_tcp(addr: &SocketAddr) -> Result<Socket> {
    init();

    let sock = try!(SocketImp::new(addr, libc::SOCK_STREAM));

    let (addrp, len) = sockaddr(addr);
    try!(cvt_r(|| unsafe { libc::connect(*sock.as_inner(), addrp, len) }));
    Ok(Socket(sock))
}

pub fn bind_tcp(addr: &SocketAddr) -> Result<Socket> {
    init();

    let sock = try!(SocketImp::new(addr, libc::SOCK_STREAM));

    // On platforms with Berkeley-derived sockets, this allows
    // to quickly rebind a socket, without needing to wait for
    // the OS to clean up the previous one.
    if !cfg!(windows) {
        try!(setsockopt(&sock, libc::SOL_SOCKET, libc::SO_REUSEADDR,
                        1 as c_int));
    }

    // Bind our new socket
    let (addrp, len) = sockaddr(addr);
    try!(cvt(unsafe { libc::bind(*sock.as_inner(), addrp, len) }));

    // Start listening
    try!(cvt(unsafe { libc::listen(*sock.as_inner(), 128) }));
    Ok(Socket(sock))
}

pub fn bind_udp(addr: &SocketAddr) -> Result<Socket> {
    init();

    let sock = try!(SocketImp::new(addr, libc::SOCK_DGRAM));
    let (addrp, len) = sockaddr(addr);
    try!(cvt(unsafe { libc::bind(*sock.as_inner(), addrp, len) }));
    Ok(Socket(sock))
}
