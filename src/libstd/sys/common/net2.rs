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

use ffi::CString;
use io::{self, Error, ErrorKind};
use libc::{self, c_int, c_char, c_void, socklen_t};
use mem;
use net::{IpAddr, SocketAddr, Shutdown};
use num::Int;
use sys::c;
use sys::net::{cvt, cvt_r, cvt_gai, Socket, init, wrlen_t};
use sys_common::{AsInner, FromInner, IntoInner};

////////////////////////////////////////////////////////////////////////////////
// sockaddr and misc bindings
////////////////////////////////////////////////////////////////////////////////

fn hton<I: Int>(i: I) -> I { i.to_be() }
fn ntoh<I: Int>(i: I) -> I { Int::from_be(i) }

fn setsockopt<T>(sock: &Socket, opt: c_int, val: c_int,
                     payload: T) -> io::Result<()> {
    unsafe {
        let payload = &payload as *const T as *const c_void;
        try!(cvt(libc::setsockopt(*sock.as_inner(), opt, val, payload,
                                  mem::size_of::<T>() as socklen_t)));
        Ok(())
    }
}

#[allow(dead_code)]
fn getsockopt<T: Copy>(sock: &Socket, opt: c_int,
                           val: c_int) -> io::Result<T> {
    unsafe {
        let mut slot: T = mem::zeroed();
        let mut len = mem::size_of::<T>() as socklen_t;
        let ret = try!(cvt(c::getsockopt(*sock.as_inner(), opt, val,
                                         &mut slot as *mut _ as *mut _,
                                         &mut len)));
        assert_eq!(ret as usize, mem::size_of::<T>());
        Ok(slot)
    }
}

fn sockname<F>(f: F) -> io::Result<SocketAddr>
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
                    len: usize) -> io::Result<SocketAddr> {
    match storage.ss_family as libc::c_int {
        libc::AF_INET => {
            assert!(len as usize >= mem::size_of::<libc::sockaddr_in>());
            Ok(FromInner::from_inner(unsafe {
                *(storage as *const _ as *const libc::sockaddr_in)
            }))
        }
        libc::AF_INET6 => {
            assert!(len as usize >= mem::size_of::<libc::sockaddr_in6>());
            Ok(FromInner::from_inner(unsafe {
                *(storage as *const _ as *const libc::sockaddr_in6)
            }))
        }
        _ => {
            Err(Error::new(ErrorKind::InvalidInput, "invalid argument", None))
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// get_host_addresses
////////////////////////////////////////////////////////////////////////////////

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
    type Item = io::Result<SocketAddr>;
    fn next(&mut self) -> Option<io::Result<SocketAddr>> {
        unsafe {
            if self.cur.is_null() { return None }
            let ret = sockaddr_to_addr(mem::transmute((*self.cur).ai_addr),
                                       (*self.cur).ai_addrlen as usize);
            self.cur = (*self.cur).ai_next as *mut libc::addrinfo;
            Some(ret)
        }
    }
}

impl Drop for LookupHost {
    fn drop(&mut self) {
        unsafe { freeaddrinfo(self.original) }
    }
}

pub fn lookup_host(host: &str) -> io::Result<LookupHost> {
    init();

    let c_host = try!(CString::new(host));
    let mut res = 0 as *mut _;
    unsafe {
        try!(cvt_gai(getaddrinfo(c_host.as_ptr(), 0 as *const _, 0 as *const _,
                                 &mut res)));
        Ok(LookupHost { original: res, cur: res })
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

        let sock = try!(Socket::new(addr, libc::SOCK_STREAM));

        let (addrp, len) = addr.into_inner();
        try!(cvt_r(|| unsafe { libc::connect(*sock.as_inner(), addrp, len) }));
        Ok(TcpStream { inner: sock })
    }

    pub fn socket(&self) -> &Socket { &self.inner }

    pub fn set_nodelay(&self, nodelay: bool) -> io::Result<()> {
        setsockopt(&self.inner, libc::IPPROTO_TCP, libc::TCP_NODELAY,
                   nodelay as c_int)
    }

    pub fn set_keepalive(&self, seconds: Option<u32>) -> io::Result<()> {
        let ret = setsockopt(&self.inner, libc::SOL_SOCKET, libc::SO_KEEPALIVE,
                             seconds.is_some() as c_int);
        match seconds {
            Some(n) => ret.and_then(|()| self.set_tcp_keepalive(n)),
            None => ret,
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn set_tcp_keepalive(&self, seconds: u32) -> io::Result<()> {
        setsockopt(&self.inner, libc::IPPROTO_TCP, libc::TCP_KEEPALIVE,
                   seconds as c_int)
    }
    #[cfg(any(target_os = "freebsd", target_os = "dragonfly"))]
    fn set_tcp_keepalive(&self, seconds: u32) -> io::Result<()> {
        setsockopt(&self.inner, libc::IPPROTO_TCP, libc::TCP_KEEPIDLE,
                   seconds as c_int)
    }
    #[cfg(not(any(target_os = "macos",
                  target_os = "ios",
                  target_os = "freebsd",
                  target_os = "dragonfly")))]
    fn set_tcp_keepalive(&self, _seconds: u32) -> io::Result<()> {
        Ok(())
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let ret = try!(cvt(unsafe {
            libc::send(*self.inner.as_inner(),
                       buf.as_ptr() as *const c_void,
                       buf.len() as wrlen_t,
                       0)
        }));
        Ok(ret as usize)
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        sockname(|buf, len| unsafe {
            libc::getpeername(*self.inner.as_inner(), buf, len)
        })
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        sockname(|buf, len| unsafe {
            libc::getsockname(*self.inner.as_inner(), buf, len)
        })
    }

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        use libc::consts::os::bsd44::SHUT_RDWR;

        let how = match how {
            Shutdown::Write => libc::SHUT_WR,
            Shutdown::Read => libc::SHUT_RD,
            Shutdown::Both => SHUT_RDWR,
        };
        try!(cvt(unsafe { libc::shutdown(*self.inner.as_inner(), how) }));
        Ok(())
    }

    pub fn duplicate(&self) -> io::Result<TcpStream> {
        self.inner.duplicate().map(|s| TcpStream { inner: s })
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

        let sock = try!(Socket::new(addr, libc::SOCK_STREAM));

        // On platforms with Berkeley-derived sockets, this allows
        // to quickly rebind a socket, without needing to wait for
        // the OS to clean up the previous one.
        if !cfg!(windows) {
            try!(setsockopt(&sock, libc::SOL_SOCKET, libc::SO_REUSEADDR,
                            1 as c_int));
        }

        // Bind our new socket
        let (addrp, len) = addr.into_inner();
        try!(cvt(unsafe { libc::bind(*sock.as_inner(), addrp, len) }));

        // Start listening
        try!(cvt(unsafe { libc::listen(*sock.as_inner(), 128) }));
        Ok(TcpListener { inner: sock })
    }

    pub fn socket(&self) -> &Socket { &self.inner }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        sockname(|buf, len| unsafe {
            libc::getsockname(*self.inner.as_inner(), buf, len)
        })
    }

    pub fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        let mut storage: libc::sockaddr_storage = unsafe { mem::zeroed() };
        let mut len = mem::size_of_val(&storage) as socklen_t;
        let sock = try!(self.inner.accept(&mut storage as *mut _ as *mut _,
                                          &mut len));
        let addr = try!(sockaddr_to_addr(&storage, len as usize));
        Ok((TcpStream { inner: sock, }, addr))
    }

    pub fn duplicate(&self) -> io::Result<TcpListener> {
        self.inner.duplicate().map(|s| TcpListener { inner: s })
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

        let sock = try!(Socket::new(addr, libc::SOCK_DGRAM));
        let (addrp, len) = addr.into_inner();
        try!(cvt(unsafe { libc::bind(*sock.as_inner(), addrp, len) }));
        Ok(UdpSocket { inner: sock })
    }

    pub fn socket(&self) -> &Socket { &self.inner }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        sockname(|buf, len| unsafe {
            libc::getsockname(*self.inner.as_inner(), buf, len)
        })
    }

    pub fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        let mut storage: libc::sockaddr_storage = unsafe { mem::zeroed() };
        let mut addrlen = mem::size_of_val(&storage) as socklen_t;

        let n = try!(cvt(unsafe {
            libc::recvfrom(*self.inner.as_inner(),
                           buf.as_mut_ptr() as *mut c_void,
                           buf.len() as wrlen_t, 0,
                           &mut storage as *mut _ as *mut _, &mut addrlen)
        }));
        Ok((n as usize, try!(sockaddr_to_addr(&storage, addrlen as usize))))
    }

    pub fn send_to(&self, buf: &[u8], dst: &SocketAddr) -> io::Result<usize> {
        let (dstp, dstlen) = dst.into_inner();
        let ret = try!(cvt(unsafe {
            libc::sendto(*self.inner.as_inner(),
                         buf.as_ptr() as *const c_void, buf.len() as wrlen_t,
                         0, dstp, dstlen)
        }));
        Ok(ret as usize)
    }

    pub fn set_broadcast(&self, on: bool) -> io::Result<()> {
        setsockopt(&self.inner, libc::SOL_SOCKET, libc::SO_BROADCAST,
                   on as c_int)
    }

    pub fn set_multicast_loop(&self, on: bool) -> io::Result<()> {
        setsockopt(&self.inner, libc::IPPROTO_IP,
                   libc::IP_MULTICAST_LOOP, on as c_int)
    }

    pub fn join_multicast(&self, multi: &IpAddr) -> io::Result<()> {
        match *multi {
            IpAddr::V4(..) => {
                self.set_membership(multi, libc::IP_ADD_MEMBERSHIP)
            }
            IpAddr::V6(..) => {
                self.set_membership(multi, libc::IPV6_ADD_MEMBERSHIP)
            }
        }
    }
    pub fn leave_multicast(&self, multi: &IpAddr) -> io::Result<()> {
        match *multi {
            IpAddr::V4(..) => {
                self.set_membership(multi, libc::IP_DROP_MEMBERSHIP)
            }
            IpAddr::V6(..) => {
                self.set_membership(multi, libc::IPV6_DROP_MEMBERSHIP)
            }
        }
    }
    fn set_membership(&self, addr: &IpAddr, opt: c_int) -> io::Result<()> {
        match *addr {
            IpAddr::V4(ref addr) => {
                let mreq = libc::ip_mreq {
                    imr_multiaddr: *addr.as_inner(),
                    // interface == INADDR_ANY
                    imr_interface: libc::in_addr { s_addr: 0x0 },
                };
                setsockopt(&self.inner, libc::IPPROTO_IP, opt, mreq)
            }
            IpAddr::V6(ref addr) => {
                let mreq = libc::ip6_mreq {
                    ipv6mr_multiaddr: *addr.as_inner(),
                    ipv6mr_interface: 0,
                };
                setsockopt(&self.inner, libc::IPPROTO_IPV6, opt, mreq)
            }
        }
    }

    pub fn multicast_time_to_live(&self, ttl: i32) -> io::Result<()> {
        setsockopt(&self.inner, libc::IPPROTO_IP, libc::IP_MULTICAST_TTL,
                   ttl as c_int)
    }

    pub fn time_to_live(&self, ttl: i32) -> io::Result<()> {
        setsockopt(&self.inner, libc::IPPROTO_IP, libc::IP_TTL, ttl as c_int)
    }

    pub fn duplicate(&self) -> io::Result<UdpSocket> {
        self.inner.duplicate().map(|s| UdpSocket { inner: s })
    }
}
