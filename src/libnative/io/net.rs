// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc;
use std::cast;
use std::io::net::ip;
use std::io;
use std::mem;
use std::rt::rtio;
use std::sync::arc::UnsafeArc;

use super::{IoResult, retry, keep_going};
use super::c;
use super::util;

////////////////////////////////////////////////////////////////////////////////
// sockaddr and misc bindings
////////////////////////////////////////////////////////////////////////////////

#[cfg(windows)] pub type sock_t = libc::SOCKET;
#[cfg(unix)]    pub type sock_t = super::file::fd_t;

pub fn htons(u: u16) -> u16 {
    mem::to_be16(u)
}
pub fn ntohs(u: u16) -> u16 {
    mem::from_be16(u)
}

enum InAddr {
    InAddr(libc::in_addr),
    In6Addr(libc::in6_addr),
}

fn ip_to_inaddr(ip: ip::IpAddr) -> InAddr {
    match ip {
        ip::Ipv4Addr(a, b, c, d) => {
            InAddr(libc::in_addr {
                s_addr: (d as u32 << 24) |
                        (c as u32 << 16) |
                        (b as u32 <<  8) |
                        (a as u32 <<  0)
            })
        }
        ip::Ipv6Addr(a, b, c, d, e, f, g, h) => {
            In6Addr(libc::in6_addr {
                s6_addr: [
                    htons(a),
                    htons(b),
                    htons(c),
                    htons(d),
                    htons(e),
                    htons(f),
                    htons(g),
                    htons(h),
                ]
            })
        }
    }
}

fn addr_to_sockaddr(addr: ip::SocketAddr) -> (libc::sockaddr_storage, uint) {
    unsafe {
        let storage: libc::sockaddr_storage = mem::init();
        let len = match ip_to_inaddr(addr.ip) {
            InAddr(inaddr) => {
                let storage: *mut libc::sockaddr_in = cast::transmute(&storage);
                (*storage).sin_family = libc::AF_INET as libc::sa_family_t;
                (*storage).sin_port = htons(addr.port);
                (*storage).sin_addr = inaddr;
                mem::size_of::<libc::sockaddr_in>()
            }
            In6Addr(inaddr) => {
                let storage: *mut libc::sockaddr_in6 = cast::transmute(&storage);
                (*storage).sin6_family = libc::AF_INET6 as libc::sa_family_t;
                (*storage).sin6_port = htons(addr.port);
                (*storage).sin6_addr = inaddr;
                mem::size_of::<libc::sockaddr_in6>()
            }
        };
        return (storage, len);
    }
}

fn socket(addr: ip::SocketAddr, ty: libc::c_int) -> IoResult<sock_t> {
    unsafe {
        let fam = match addr.ip {
            ip::Ipv4Addr(..) => libc::AF_INET,
            ip::Ipv6Addr(..) => libc::AF_INET6,
        };
        match libc::socket(fam, ty, 0) {
            -1 => Err(super::last_error()),
            fd => Ok(fd),
        }
    }
}

fn setsockopt<T>(fd: sock_t, opt: libc::c_int, val: libc::c_int,
                 payload: T) -> IoResult<()> {
    unsafe {
        let payload = &payload as *T as *libc::c_void;
        let ret = libc::setsockopt(fd, opt, val,
                                   payload,
                                   mem::size_of::<T>() as libc::socklen_t);
        if ret != 0 {
            Err(last_error())
        } else {
            Ok(())
        }
    }
}

pub fn getsockopt<T: Copy>(fd: sock_t, opt: libc::c_int,
                           val: libc::c_int) -> IoResult<T> {
    unsafe {
        let mut slot: T = mem::init();
        let mut len = mem::size_of::<T>() as libc::socklen_t;
        let ret = c::getsockopt(fd, opt, val,
                                &mut slot as *mut _ as *mut _,
                                &mut len);
        if ret != 0 {
            Err(last_error())
        } else {
            assert!(len as uint == mem::size_of::<T>());
            Ok(slot)
        }
    }
}

#[cfg(windows)]
fn last_error() -> io::IoError {
    io::IoError::from_errno(unsafe { c::WSAGetLastError() } as uint, true)
}

#[cfg(not(windows))]
fn last_error() -> io::IoError {
    super::last_error()
}

#[cfg(windows)] unsafe fn close(sock: sock_t) { let _ = libc::closesocket(sock); }
#[cfg(unix)]    unsafe fn close(sock: sock_t) { let _ = libc::close(sock); }

fn sockname(fd: sock_t,
            f: extern "system" unsafe fn(sock_t, *mut libc::sockaddr,
                                         *mut libc::socklen_t) -> libc::c_int)
    -> IoResult<ip::SocketAddr>
{
    let mut storage: libc::sockaddr_storage = unsafe { mem::init() };
    let mut len = mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    unsafe {
        let storage = &mut storage as *mut libc::sockaddr_storage;
        let ret = f(fd,
                    storage as *mut libc::sockaddr,
                    &mut len as *mut libc::socklen_t);
        if ret != 0 {
            return Err(last_error())
        }
    }
    return sockaddr_to_addr(&storage, len as uint);
}

pub fn sockaddr_to_addr(storage: &libc::sockaddr_storage,
                        len: uint) -> IoResult<ip::SocketAddr> {
    match storage.ss_family as libc::c_int {
        libc::AF_INET => {
            assert!(len as uint >= mem::size_of::<libc::sockaddr_in>());
            let storage: &libc::sockaddr_in = unsafe {
                cast::transmute(storage)
            };
            let addr = storage.sin_addr.s_addr as u32;
            let a = (addr >>  0) as u8;
            let b = (addr >>  8) as u8;
            let c = (addr >> 16) as u8;
            let d = (addr >> 24) as u8;
            Ok(ip::SocketAddr {
                ip: ip::Ipv4Addr(a, b, c, d),
                port: ntohs(storage.sin_port),
            })
        }
        libc::AF_INET6 => {
            assert!(len as uint >= mem::size_of::<libc::sockaddr_in6>());
            let storage: &libc::sockaddr_in6 = unsafe {
                cast::transmute(storage)
            };
            let a = ntohs(storage.sin6_addr.s6_addr[0]);
            let b = ntohs(storage.sin6_addr.s6_addr[1]);
            let c = ntohs(storage.sin6_addr.s6_addr[2]);
            let d = ntohs(storage.sin6_addr.s6_addr[3]);
            let e = ntohs(storage.sin6_addr.s6_addr[4]);
            let f = ntohs(storage.sin6_addr.s6_addr[5]);
            let g = ntohs(storage.sin6_addr.s6_addr[6]);
            let h = ntohs(storage.sin6_addr.s6_addr[7]);
            Ok(ip::SocketAddr {
                ip: ip::Ipv6Addr(a, b, c, d, e, f, g, h),
                port: ntohs(storage.sin6_port),
            })
        }
        _ => {
            Err(io::standard_error(io::OtherIoError))
        }
    }
}

#[cfg(unix)]
pub fn init() {}

#[cfg(windows)]
pub fn init() {

    unsafe {
        use std::unstable::mutex::{StaticNativeMutex, NATIVE_MUTEX_INIT};
        static mut INITIALIZED: bool = false;
        static mut LOCK: StaticNativeMutex = NATIVE_MUTEX_INIT;

        let _guard = LOCK.lock();
        if !INITIALIZED {
            let mut data: c::WSADATA = mem::init();
            let ret = c::WSAStartup(0x202,      // version 2.2
                                    &mut data);
            assert_eq!(ret, 0);
            INITIALIZED = true;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// TCP streams
////////////////////////////////////////////////////////////////////////////////

pub struct TcpStream {
    inner: UnsafeArc<Inner>,
}

struct Inner {
    fd: sock_t,
}

impl TcpStream {
    pub fn connect(addr: ip::SocketAddr,
                   timeout: Option<u64>) -> IoResult<TcpStream> {
        let fd = try!(socket(addr, libc::SOCK_STREAM));
        let (addr, len) = addr_to_sockaddr(addr);
        let inner = Inner { fd: fd };
        let ret = TcpStream { inner: UnsafeArc::new(inner) };

        let len = len as libc::socklen_t;
        let addrp = &addr as *_ as *libc::sockaddr;
        match timeout {
            Some(timeout) => {
                try!(util::connect_timeout(fd, addrp, len, timeout));
                Ok(ret)
            },
            None => {
                match retry(|| unsafe { libc::connect(fd, addrp, len) }) {
                    -1 => Err(last_error()),
                    _ => Ok(ret),
                }
            }
        }
    }

    pub fn fd(&self) -> sock_t {
        // This unsafety is fine because it's just a read-only arc
        unsafe { (*self.inner.get()).fd }
    }

    fn set_nodelay(&mut self, nodelay: bool) -> IoResult<()> {
        setsockopt(self.fd(), libc::IPPROTO_TCP, libc::TCP_NODELAY,
                   nodelay as libc::c_int)
    }

    fn set_keepalive(&mut self, seconds: Option<uint>) -> IoResult<()> {
        let ret = setsockopt(self.fd(), libc::SOL_SOCKET, libc::SO_KEEPALIVE,
                             seconds.is_some() as libc::c_int);
        match seconds {
            Some(n) => ret.and_then(|()| self.set_tcp_keepalive(n)),
            None => ret,
        }
    }

    #[cfg(target_os = "macos")]
    fn set_tcp_keepalive(&mut self, seconds: uint) -> IoResult<()> {
        setsockopt(self.fd(), libc::IPPROTO_TCP, libc::TCP_KEEPALIVE,
                   seconds as libc::c_int)
    }
    #[cfg(target_os = "freebsd")]
    fn set_tcp_keepalive(&mut self, seconds: uint) -> IoResult<()> {
        setsockopt(self.fd(), libc::IPPROTO_TCP, libc::TCP_KEEPIDLE,
                   seconds as libc::c_int)
    }
    #[cfg(not(target_os = "macos"), not(target_os = "freebsd"))]
    fn set_tcp_keepalive(&mut self, _seconds: uint) -> IoResult<()> {
        Ok(())
    }
}

#[cfg(windows)] type wrlen = libc::c_int;
#[cfg(not(windows))] type wrlen = libc::size_t;

impl rtio::RtioTcpStream for TcpStream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        let ret = retry(|| {
            unsafe {
                libc::recv(self.fd(),
                           buf.as_mut_ptr() as *mut libc::c_void,
                           buf.len() as wrlen,
                           0) as libc::c_int
            }
        });
        if ret == 0 {
            Err(io::standard_error(io::EndOfFile))
        } else if ret < 0 {
            Err(last_error())
        } else {
            Ok(ret as uint)
        }
    }
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        let ret = keep_going(buf, |buf, len| unsafe {
            libc::send(self.fd(),
                       buf as *mut libc::c_void,
                       len as wrlen,
                       0) as i64
        });
        if ret < 0 {
            Err(super::last_error())
        } else {
            Ok(())
        }
    }
    fn peer_name(&mut self) -> IoResult<ip::SocketAddr> {
        sockname(self.fd(), libc::getpeername)
    }
    fn control_congestion(&mut self) -> IoResult<()> {
        self.set_nodelay(false)
    }
    fn nodelay(&mut self) -> IoResult<()> {
        self.set_nodelay(true)
    }
    fn keepalive(&mut self, delay_in_seconds: uint) -> IoResult<()> {
        self.set_keepalive(Some(delay_in_seconds))
    }
    fn letdie(&mut self) -> IoResult<()> {
        self.set_keepalive(None)
    }

    fn clone(&self) -> Box<rtio::RtioTcpStream:Send> {
        box TcpStream {
            inner: self.inner.clone(),
        } as Box<rtio::RtioTcpStream:Send>
    }
    fn close_write(&mut self) -> IoResult<()> {
        super::mkerr_libc(unsafe {
            libc::shutdown(self.fd(), libc::SHUT_WR)
        })
    }
}

impl rtio::RtioSocket for TcpStream {
    fn socket_name(&mut self) -> IoResult<ip::SocketAddr> {
        sockname(self.fd(), libc::getsockname)
    }
}

impl Drop for Inner {
    fn drop(&mut self) { unsafe { close(self.fd); } }
}

////////////////////////////////////////////////////////////////////////////////
// TCP listeners
////////////////////////////////////////////////////////////////////////////////

pub struct TcpListener {
    inner: Inner,
}

impl TcpListener {
    pub fn bind(addr: ip::SocketAddr) -> IoResult<TcpListener> {
        unsafe {
            socket(addr, libc::SOCK_STREAM).and_then(|fd| {
                let (addr, len) = addr_to_sockaddr(addr);
                let addrp = &addr as *libc::sockaddr_storage;
                let inner = Inner { fd: fd };
                let ret = TcpListener { inner: inner };
                // On platforms with Berkeley-derived sockets, this allows
                // to quickly rebind a socket, without needing to wait for
                // the OS to clean up the previous one.
                if cfg!(unix) {
                    match setsockopt(fd, libc::SOL_SOCKET,
                                     libc::SO_REUSEADDR,
                                     1 as libc::c_int) {
                        Err(n) => { return Err(n); },
                        Ok(..) => { }
                    }
                }
                match libc::bind(fd, addrp as *libc::sockaddr,
                                 len as libc::socklen_t) {
                    -1 => Err(last_error()),
                    _ => Ok(ret),
                }
            })
        }
    }

    pub fn fd(&self) -> sock_t { self.inner.fd }

    pub fn native_listen(self, backlog: int) -> IoResult<TcpAcceptor> {
        match unsafe { libc::listen(self.fd(), backlog as libc::c_int) } {
            -1 => Err(last_error()),
            _ => Ok(TcpAcceptor { listener: self, deadline: 0 })
        }
    }
}

impl rtio::RtioTcpListener for TcpListener {
    fn listen(~self) -> IoResult<Box<rtio::RtioTcpAcceptor:Send>> {
        self.native_listen(128).map(|a| {
            box a as Box<rtio::RtioTcpAcceptor:Send>
        })
    }
}

impl rtio::RtioSocket for TcpListener {
    fn socket_name(&mut self) -> IoResult<ip::SocketAddr> {
        sockname(self.fd(), libc::getsockname)
    }
}

pub struct TcpAcceptor {
    listener: TcpListener,
    deadline: u64,
}

impl TcpAcceptor {
    pub fn fd(&self) -> sock_t { self.listener.fd() }

    pub fn native_accept(&mut self) -> IoResult<TcpStream> {
        if self.deadline != 0 {
            try!(util::accept_deadline(self.fd(), self.deadline));
        }
        unsafe {
            let mut storage: libc::sockaddr_storage = mem::init();
            let storagep = &mut storage as *mut libc::sockaddr_storage;
            let size = mem::size_of::<libc::sockaddr_storage>();
            let mut size = size as libc::socklen_t;
            match retry(|| {
                libc::accept(self.fd(),
                             storagep as *mut libc::sockaddr,
                             &mut size as *mut libc::socklen_t) as libc::c_int
            }) as sock_t {
                -1 => Err(last_error()),
                fd => Ok(TcpStream { inner: UnsafeArc::new(Inner { fd: fd })})
            }
        }
    }
}

impl rtio::RtioSocket for TcpAcceptor {
    fn socket_name(&mut self) -> IoResult<ip::SocketAddr> {
        sockname(self.fd(), libc::getsockname)
    }
}

impl rtio::RtioTcpAcceptor for TcpAcceptor {
    fn accept(&mut self) -> IoResult<Box<rtio::RtioTcpStream:Send>> {
        self.native_accept().map(|s| box s as Box<rtio::RtioTcpStream:Send>)
    }

    fn accept_simultaneously(&mut self) -> IoResult<()> { Ok(()) }
    fn dont_accept_simultaneously(&mut self) -> IoResult<()> { Ok(()) }
    fn set_timeout(&mut self, timeout: Option<u64>) {
        self.deadline = timeout.map(|a| ::io::timer::now() + a).unwrap_or(0);
    }
}

////////////////////////////////////////////////////////////////////////////////
// UDP
////////////////////////////////////////////////////////////////////////////////

pub struct UdpSocket {
    inner: UnsafeArc<Inner>,
}

impl UdpSocket {
    pub fn bind(addr: ip::SocketAddr) -> IoResult<UdpSocket> {
        unsafe {
            socket(addr, libc::SOCK_DGRAM).and_then(|fd| {
                let (addr, len) = addr_to_sockaddr(addr);
                let addrp = &addr as *libc::sockaddr_storage;
                let inner = Inner { fd: fd };
                let ret = UdpSocket { inner: UnsafeArc::new(inner) };
                match libc::bind(fd, addrp as *libc::sockaddr,
                                 len as libc::socklen_t) {
                    -1 => Err(last_error()),
                    _ => Ok(ret),
                }
            })
        }
    }

    pub fn fd(&self) -> sock_t {
        // unsafety is fine because it's just a read-only arc
        unsafe { (*self.inner.get()).fd }
    }

    pub fn set_broadcast(&mut self, on: bool) -> IoResult<()> {
        setsockopt(self.fd(), libc::SOL_SOCKET, libc::SO_BROADCAST,
                   on as libc::c_int)
    }

    pub fn set_multicast_loop(&mut self, on: bool) -> IoResult<()> {
        setsockopt(self.fd(), libc::IPPROTO_IP, libc::IP_MULTICAST_LOOP,
                   on as libc::c_int)
    }

    pub fn set_membership(&mut self, addr: ip::IpAddr,
                          opt: libc::c_int) -> IoResult<()> {
        match ip_to_inaddr(addr) {
            InAddr(addr) => {
                let mreq = libc::ip_mreq {
                    imr_multiaddr: addr,
                    // interface == INADDR_ANY
                    imr_interface: libc::in_addr { s_addr: 0x0 },
                };
                setsockopt(self.fd(), libc::IPPROTO_IP, opt, mreq)
            }
            In6Addr(addr) => {
                let mreq = libc::ip6_mreq {
                    ipv6mr_multiaddr: addr,
                    ipv6mr_interface: 0,
                };
                setsockopt(self.fd(), libc::IPPROTO_IPV6, opt, mreq)
            }
        }
    }
}

impl rtio::RtioSocket for UdpSocket {
    fn socket_name(&mut self) -> IoResult<ip::SocketAddr> {
        sockname(self.fd(), libc::getsockname)
    }
}

#[cfg(windows)] type msglen_t = libc::c_int;
#[cfg(unix)]    type msglen_t = libc::size_t;

impl rtio::RtioUdpSocket for UdpSocket {
    fn recvfrom(&mut self, buf: &mut [u8]) -> IoResult<(uint, ip::SocketAddr)> {
        unsafe {
            let mut storage: libc::sockaddr_storage = mem::init();
            let storagep = &mut storage as *mut libc::sockaddr_storage;
            let mut addrlen: libc::socklen_t =
                    mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
            let ret = retry(|| {
                libc::recvfrom(self.fd(),
                               buf.as_ptr() as *mut libc::c_void,
                               buf.len() as msglen_t,
                               0,
                               storagep as *mut libc::sockaddr,
                               &mut addrlen) as libc::c_int
            });
            if ret < 0 { return Err(last_error()) }
            sockaddr_to_addr(&storage, addrlen as uint).and_then(|addr| {
                Ok((ret as uint, addr))
            })
        }
    }
    fn sendto(&mut self, buf: &[u8], dst: ip::SocketAddr) -> IoResult<()> {
        let (dst, len) = addr_to_sockaddr(dst);
        let dstp = &dst as *libc::sockaddr_storage;
        unsafe {
            let ret = retry(|| {
                libc::sendto(self.fd(),
                             buf.as_ptr() as *libc::c_void,
                             buf.len() as msglen_t,
                             0,
                             dstp as *libc::sockaddr,
                             len as libc::socklen_t) as libc::c_int
            });
            match ret {
                -1 => Err(last_error()),
                n if n as uint != buf.len() => {
                    Err(io::IoError {
                        kind: io::OtherIoError,
                        desc: "couldn't send entire packet at once",
                        detail: None,
                    })
                }
                _ => Ok(())
            }
        }
    }

    fn join_multicast(&mut self, multi: ip::IpAddr) -> IoResult<()> {
        match multi {
            ip::Ipv4Addr(..) => {
                self.set_membership(multi, libc::IP_ADD_MEMBERSHIP)
            }
            ip::Ipv6Addr(..) => {
                self.set_membership(multi, libc::IPV6_ADD_MEMBERSHIP)
            }
        }
    }
    fn leave_multicast(&mut self, multi: ip::IpAddr) -> IoResult<()> {
        match multi {
            ip::Ipv4Addr(..) => {
                self.set_membership(multi, libc::IP_DROP_MEMBERSHIP)
            }
            ip::Ipv6Addr(..) => {
                self.set_membership(multi, libc::IPV6_DROP_MEMBERSHIP)
            }
        }
    }

    fn loop_multicast_locally(&mut self) -> IoResult<()> {
        self.set_multicast_loop(true)
    }
    fn dont_loop_multicast_locally(&mut self) -> IoResult<()> {
        self.set_multicast_loop(false)
    }

    fn multicast_time_to_live(&mut self, ttl: int) -> IoResult<()> {
        setsockopt(self.fd(), libc::IPPROTO_IP, libc::IP_MULTICAST_TTL,
                   ttl as libc::c_int)
    }
    fn time_to_live(&mut self, ttl: int) -> IoResult<()> {
        setsockopt(self.fd(), libc::IPPROTO_IP, libc::IP_TTL, ttl as libc::c_int)
    }

    fn hear_broadcasts(&mut self) -> IoResult<()> {
        self.set_broadcast(true)
    }
    fn ignore_broadcasts(&mut self) -> IoResult<()> {
        self.set_broadcast(false)
    }

    fn clone(&self) -> Box<rtio::RtioUdpSocket:Send> {
        box UdpSocket {
            inner: self.inner.clone(),
        } as Box<rtio::RtioUdpSocket:Send>
    }
}
