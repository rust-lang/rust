// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use alloc::arc::Arc;
use libc;
use std::mem;
use std::rt::mutex;
use std::rt::rtio;
use std::rt::rtio::{IoResult, IoError};

use super::{retry, keep_going};
use super::c;
use super::util;

////////////////////////////////////////////////////////////////////////////////
// sockaddr and misc bindings
////////////////////////////////////////////////////////////////////////////////

#[cfg(windows)] pub type sock_t = libc::SOCKET;
#[cfg(unix)]    pub type sock_t = super::file::fd_t;

pub fn htons(u: u16) -> u16 {
    u.to_be()
}
pub fn ntohs(u: u16) -> u16 {
    Int::from_be(u)
}

enum InAddr {
    InAddr(libc::in_addr),
    In6Addr(libc::in6_addr),
}

fn ip_to_inaddr(ip: rtio::IpAddr) -> InAddr {
    match ip {
        rtio::Ipv4Addr(a, b, c, d) => {
            let ip = (a as u32 << 24) |
                     (b as u32 << 16) |
                     (c as u32 <<  8) |
                     (d as u32 <<  0);
            InAddr(libc::in_addr {
                s_addr: Int::from_be(ip)
            })
        }
        rtio::Ipv6Addr(a, b, c, d, e, f, g, h) => {
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

fn addr_to_sockaddr(addr: rtio::SocketAddr,
                    storage: &mut libc::sockaddr_storage)
                    -> libc::socklen_t {
    unsafe {
        let len = match ip_to_inaddr(addr.ip) {
            InAddr(inaddr) => {
                let storage = storage as *mut _ as *mut libc::sockaddr_in;
                (*storage).sin_family = libc::AF_INET as libc::sa_family_t;
                (*storage).sin_port = htons(addr.port);
                (*storage).sin_addr = inaddr;
                mem::size_of::<libc::sockaddr_in>()
            }
            In6Addr(inaddr) => {
                let storage = storage as *mut _ as *mut libc::sockaddr_in6;
                (*storage).sin6_family = libc::AF_INET6 as libc::sa_family_t;
                (*storage).sin6_port = htons(addr.port);
                (*storage).sin6_addr = inaddr;
                mem::size_of::<libc::sockaddr_in6>()
            }
        };
        return len as libc::socklen_t;
    }
}

fn socket(addr: rtio::SocketAddr, ty: libc::c_int) -> IoResult<sock_t> {
    unsafe {
        let fam = match addr.ip {
            rtio::Ipv4Addr(..) => libc::AF_INET,
            rtio::Ipv6Addr(..) => libc::AF_INET6,
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
        let payload = &payload as *const T as *const libc::c_void;
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
        let mut slot: T = mem::zeroed();
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
pub fn last_error() -> IoError {
    use std::os;
    let code = unsafe { c::WSAGetLastError() as uint };
    IoError {
        code: code,
        extra: 0,
        detail: Some(os::error_string(code)),
    }
}

#[cfg(not(windows))]
fn last_error() -> IoError {
    super::last_error()
}

#[cfg(windows)] unsafe fn close(sock: sock_t) { let _ = libc::closesocket(sock); }
#[cfg(unix)]    unsafe fn close(sock: sock_t) { let _ = libc::close(sock); }

fn sockname(fd: sock_t,
            f: unsafe extern "system" fn(sock_t, *mut libc::sockaddr,
                                         *mut libc::socklen_t) -> libc::c_int)
    -> IoResult<rtio::SocketAddr>
{
    let mut storage: libc::sockaddr_storage = unsafe { mem::zeroed() };
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
                        len: uint) -> IoResult<rtio::SocketAddr> {
    match storage.ss_family as libc::c_int {
        libc::AF_INET => {
            assert!(len as uint >= mem::size_of::<libc::sockaddr_in>());
            let storage: &libc::sockaddr_in = unsafe {
                mem::transmute(storage)
            };
            let ip = (storage.sin_addr.s_addr as u32).to_be();
            let a = (ip >> 24) as u8;
            let b = (ip >> 16) as u8;
            let c = (ip >>  8) as u8;
            let d = (ip >>  0) as u8;
            Ok(rtio::SocketAddr {
                ip: rtio::Ipv4Addr(a, b, c, d),
                port: ntohs(storage.sin_port),
            })
        }
        libc::AF_INET6 => {
            assert!(len as uint >= mem::size_of::<libc::sockaddr_in6>());
            let storage: &libc::sockaddr_in6 = unsafe {
                mem::transmute(storage)
            };
            let a = ntohs(storage.sin6_addr.s6_addr[0]);
            let b = ntohs(storage.sin6_addr.s6_addr[1]);
            let c = ntohs(storage.sin6_addr.s6_addr[2]);
            let d = ntohs(storage.sin6_addr.s6_addr[3]);
            let e = ntohs(storage.sin6_addr.s6_addr[4]);
            let f = ntohs(storage.sin6_addr.s6_addr[5]);
            let g = ntohs(storage.sin6_addr.s6_addr[6]);
            let h = ntohs(storage.sin6_addr.s6_addr[7]);
            Ok(rtio::SocketAddr {
                ip: rtio::Ipv6Addr(a, b, c, d, e, f, g, h),
                port: ntohs(storage.sin6_port),
            })
        }
        _ => {
            #[cfg(unix)] use libc::EINVAL as ERROR;
            #[cfg(windows)] use libc::WSAEINVAL as ERROR;
            Err(IoError {
                code: ERROR as uint,
                extra: 0,
                detail: None,
            })
        }
    }
}

#[cfg(unix)]
pub fn init() {}

#[cfg(windows)]
pub fn init() {

    unsafe {
        use std::rt::mutex::{StaticNativeMutex, NATIVE_MUTEX_INIT};
        static mut INITIALIZED: bool = false;
        static mut LOCK: StaticNativeMutex = NATIVE_MUTEX_INIT;

        let _guard = LOCK.lock();
        if !INITIALIZED {
            let mut data: c::WSADATA = mem::zeroed();
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
    inner: Arc<Inner>,
    read_deadline: u64,
    write_deadline: u64,
}

struct Inner {
    fd: sock_t,

    // Unused on Linux, where this lock is not necessary.
    #[allow(dead_code)]
    lock: mutex::NativeMutex
}

pub struct Guard<'a> {
    pub fd: sock_t,
    pub guard: mutex::LockGuard<'a>,
}

impl Inner {
    fn new(fd: sock_t) -> Inner {
        Inner { fd: fd, lock: unsafe { mutex::NativeMutex::new() } }
    }
}

impl TcpStream {
    pub fn connect(addr: rtio::SocketAddr,
                   timeout: Option<u64>) -> IoResult<TcpStream> {
        let fd = try!(socket(addr, libc::SOCK_STREAM));
        let ret = TcpStream::new(Inner::new(fd));

        let mut storage = unsafe { mem::zeroed() };
        let len = addr_to_sockaddr(addr, &mut storage);
        let addrp = &storage as *const _ as *const libc::sockaddr;

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

    fn new(inner: Inner) -> TcpStream {
        TcpStream {
            inner: Arc::new(inner),
            read_deadline: 0,
            write_deadline: 0,
        }
    }

    pub fn fd(&self) -> sock_t { self.inner.fd }

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
    #[cfg(target_os = "ios")]
    fn set_tcp_keepalive(&mut self, seconds: uint) -> IoResult<()> {
        setsockopt(self.fd(), libc::IPPROTO_TCP, libc::TCP_KEEPALIVE,
                   seconds as libc::c_int)
    }
    #[cfg(target_os = "freebsd")]
    #[cfg(target_os = "dragonfly")]
    fn set_tcp_keepalive(&mut self, seconds: uint) -> IoResult<()> {
        setsockopt(self.fd(), libc::IPPROTO_TCP, libc::TCP_KEEPIDLE,
                   seconds as libc::c_int)
    }
    #[cfg(not(target_os = "macos"), not(target_os = "ios"), not(target_os = "freebsd"),
      not(target_os = "dragonfly"))]
    fn set_tcp_keepalive(&mut self, _seconds: uint) -> IoResult<()> {
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn lock_nonblocking(&self) {}

    #[cfg(not(target_os = "linux"))]
    fn lock_nonblocking<'a>(&'a self) -> Guard<'a> {
        let ret = Guard {
            fd: self.fd(),
            guard: unsafe { self.inner.lock.lock() },
        };
        assert!(util::set_nonblocking(self.fd(), true).is_ok());
        ret
    }
}

#[cfg(windows)] type wrlen = libc::c_int;
#[cfg(not(windows))] type wrlen = libc::size_t;

impl rtio::RtioTcpStream for TcpStream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        let fd = self.fd();
        let dolock = || self.lock_nonblocking();
        let doread = |nb| unsafe {
            let flags = if nb {c::MSG_DONTWAIT} else {0};
            libc::recv(fd,
                       buf.as_mut_ptr() as *mut libc::c_void,
                       buf.len() as wrlen,
                       flags) as libc::c_int
        };
        read(fd, self.read_deadline, dolock, doread)
    }

    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        let fd = self.fd();
        let dolock = || self.lock_nonblocking();
        let dowrite = |nb: bool, buf: *const u8, len: uint| unsafe {
            let flags = if nb {c::MSG_DONTWAIT} else {0};
            libc::send(fd,
                       buf as *mut libc::c_void,
                       len as wrlen,
                       flags) as i64
        };
        match write(fd, self.write_deadline, buf, true, dolock, dowrite) {
            Ok(_) => Ok(()),
            Err(e) => Err(e)
        }
    }
    fn peer_name(&mut self) -> IoResult<rtio::SocketAddr> {
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

    fn clone(&self) -> Box<rtio::RtioTcpStream + Send> {
        box TcpStream {
            inner: self.inner.clone(),
            read_deadline: 0,
            write_deadline: 0,
        } as Box<rtio::RtioTcpStream + Send>
    }

    fn close_write(&mut self) -> IoResult<()> {
        super::mkerr_libc(unsafe { libc::shutdown(self.fd(), libc::SHUT_WR) })
    }
    fn close_read(&mut self) -> IoResult<()> {
        super::mkerr_libc(unsafe { libc::shutdown(self.fd(), libc::SHUT_RD) })
    }

    fn set_timeout(&mut self, timeout: Option<u64>) {
        let deadline = timeout.map(|a| ::io::timer::now() + a).unwrap_or(0);
        self.read_deadline = deadline;
        self.write_deadline = deadline;
    }
    fn set_read_timeout(&mut self, timeout: Option<u64>) {
        self.read_deadline = timeout.map(|a| ::io::timer::now() + a).unwrap_or(0);
    }
    fn set_write_timeout(&mut self, timeout: Option<u64>) {
        self.write_deadline = timeout.map(|a| ::io::timer::now() + a).unwrap_or(0);
    }
}

impl rtio::RtioSocket for TcpStream {
    fn socket_name(&mut self) -> IoResult<rtio::SocketAddr> {
        sockname(self.fd(), libc::getsockname)
    }
}

impl Drop for Inner {
    fn drop(&mut self) { unsafe { close(self.fd); } }
}

#[unsafe_destructor]
impl<'a> Drop for Guard<'a> {
    fn drop(&mut self) {
        assert!(util::set_nonblocking(self.fd, false).is_ok());
    }
}

////////////////////////////////////////////////////////////////////////////////
// TCP listeners
////////////////////////////////////////////////////////////////////////////////

pub struct TcpListener {
    inner: Inner,
}

impl TcpListener {
    pub fn bind(addr: rtio::SocketAddr) -> IoResult<TcpListener> {
        let fd = try!(socket(addr, libc::SOCK_STREAM));
        let ret = TcpListener { inner: Inner::new(fd) };

        let mut storage = unsafe { mem::zeroed() };
        let len = addr_to_sockaddr(addr, &mut storage);
        let addrp = &storage as *const _ as *const libc::sockaddr;

        // On platforms with Berkeley-derived sockets, this allows
        // to quickly rebind a socket, without needing to wait for
        // the OS to clean up the previous one.
        if cfg!(unix) {
            try!(setsockopt(fd, libc::SOL_SOCKET, libc::SO_REUSEADDR,
                            1 as libc::c_int));
        }

        match unsafe { libc::bind(fd, addrp, len) } {
            -1 => Err(last_error()),
            _ => Ok(ret),
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
    fn listen(self: Box<TcpListener>)
              -> IoResult<Box<rtio::RtioTcpAcceptor + Send>> {
        self.native_listen(128).map(|a| {
            box a as Box<rtio::RtioTcpAcceptor + Send>
        })
    }
}

impl rtio::RtioSocket for TcpListener {
    fn socket_name(&mut self) -> IoResult<rtio::SocketAddr> {
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
            try!(util::await(self.fd(), Some(self.deadline), util::Readable));
        }
        unsafe {
            let mut storage: libc::sockaddr_storage = mem::zeroed();
            let storagep = &mut storage as *mut libc::sockaddr_storage;
            let size = mem::size_of::<libc::sockaddr_storage>();
            let mut size = size as libc::socklen_t;
            match retry(|| {
                libc::accept(self.fd(),
                             storagep as *mut libc::sockaddr,
                             &mut size as *mut libc::socklen_t) as libc::c_int
            }) as sock_t {
                -1 => Err(last_error()),
                fd => Ok(TcpStream::new(Inner::new(fd))),
            }
        }
    }
}

impl rtio::RtioSocket for TcpAcceptor {
    fn socket_name(&mut self) -> IoResult<rtio::SocketAddr> {
        sockname(self.fd(), libc::getsockname)
    }
}

impl rtio::RtioTcpAcceptor for TcpAcceptor {
    fn accept(&mut self) -> IoResult<Box<rtio::RtioTcpStream + Send>> {
        self.native_accept().map(|s| box s as Box<rtio::RtioTcpStream + Send>)
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
    inner: Arc<Inner>,
    read_deadline: u64,
    write_deadline: u64,
}

impl UdpSocket {
    pub fn bind(addr: rtio::SocketAddr) -> IoResult<UdpSocket> {
        let fd = try!(socket(addr, libc::SOCK_DGRAM));
        let ret = UdpSocket {
            inner: Arc::new(Inner::new(fd)),
            read_deadline: 0,
            write_deadline: 0,
        };

        let mut storage = unsafe { mem::zeroed() };
        let len = addr_to_sockaddr(addr, &mut storage);
        let addrp = &storage as *const _ as *const libc::sockaddr;

        match unsafe { libc::bind(fd, addrp, len) } {
            -1 => Err(last_error()),
            _ => Ok(ret),
        }
    }

    pub fn fd(&self) -> sock_t { self.inner.fd }

    pub fn set_broadcast(&mut self, on: bool) -> IoResult<()> {
        setsockopt(self.fd(), libc::SOL_SOCKET, libc::SO_BROADCAST,
                   on as libc::c_int)
    }

    pub fn set_multicast_loop(&mut self, on: bool) -> IoResult<()> {
        setsockopt(self.fd(), libc::IPPROTO_IP, libc::IP_MULTICAST_LOOP,
                   on as libc::c_int)
    }

    pub fn set_membership(&mut self, addr: rtio::IpAddr,
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

    #[cfg(target_os = "linux")]
    fn lock_nonblocking(&self) {}

    #[cfg(not(target_os = "linux"))]
    fn lock_nonblocking<'a>(&'a self) -> Guard<'a> {
        let ret = Guard {
            fd: self.fd(),
            guard: unsafe { self.inner.lock.lock() },
        };
        assert!(util::set_nonblocking(self.fd(), true).is_ok());
        ret
    }
}

impl rtio::RtioSocket for UdpSocket {
    fn socket_name(&mut self) -> IoResult<rtio::SocketAddr> {
        sockname(self.fd(), libc::getsockname)
    }
}

#[cfg(windows)] type msglen_t = libc::c_int;
#[cfg(unix)]    type msglen_t = libc::size_t;

impl rtio::RtioUdpSocket for UdpSocket {
    fn recv_from(&mut self, buf: &mut [u8]) -> IoResult<(uint, rtio::SocketAddr)> {
        let fd = self.fd();
        let mut storage: libc::sockaddr_storage = unsafe { mem::zeroed() };
        let storagep = &mut storage as *mut _ as *mut libc::sockaddr;
        let mut addrlen: libc::socklen_t =
                mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;

        let dolock = || self.lock_nonblocking();
        let n = try!(read(fd, self.read_deadline, dolock, |nb| unsafe {
            let flags = if nb {c::MSG_DONTWAIT} else {0};
            libc::recvfrom(fd,
                           buf.as_mut_ptr() as *mut libc::c_void,
                           buf.len() as msglen_t,
                           flags,
                           storagep,
                           &mut addrlen) as libc::c_int
        }));
        sockaddr_to_addr(&storage, addrlen as uint).and_then(|addr| {
            Ok((n as uint, addr))
        })
    }

    fn send_to(&mut self, buf: &[u8], dst: rtio::SocketAddr) -> IoResult<()> {
        let mut storage = unsafe { mem::zeroed() };
        let dstlen = addr_to_sockaddr(dst, &mut storage);
        let dstp = &storage as *const _ as *const libc::sockaddr;

        let fd = self.fd();
        let dolock = || self.lock_nonblocking();
        let dowrite = |nb, buf: *const u8, len: uint| unsafe {
            let flags = if nb {c::MSG_DONTWAIT} else {0};
            libc::sendto(fd,
                         buf as *const libc::c_void,
                         len as msglen_t,
                         flags,
                         dstp,
                         dstlen) as i64
        };

        let n = try!(write(fd, self.write_deadline, buf, false, dolock, dowrite));
        if n != buf.len() {
            Err(util::short_write(n, "couldn't send entire packet at once"))
        } else {
            Ok(())
        }
    }

    fn join_multicast(&mut self, multi: rtio::IpAddr) -> IoResult<()> {
        match multi {
            rtio::Ipv4Addr(..) => {
                self.set_membership(multi, libc::IP_ADD_MEMBERSHIP)
            }
            rtio::Ipv6Addr(..) => {
                self.set_membership(multi, libc::IPV6_ADD_MEMBERSHIP)
            }
        }
    }
    fn leave_multicast(&mut self, multi: rtio::IpAddr) -> IoResult<()> {
        match multi {
            rtio::Ipv4Addr(..) => {
                self.set_membership(multi, libc::IP_DROP_MEMBERSHIP)
            }
            rtio::Ipv6Addr(..) => {
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

    fn clone(&self) -> Box<rtio::RtioUdpSocket + Send> {
        box UdpSocket {
            inner: self.inner.clone(),
            read_deadline: 0,
            write_deadline: 0,
        } as Box<rtio::RtioUdpSocket + Send>
    }

    fn set_timeout(&mut self, timeout: Option<u64>) {
        let deadline = timeout.map(|a| ::io::timer::now() + a).unwrap_or(0);
        self.read_deadline = deadline;
        self.write_deadline = deadline;
    }
    fn set_read_timeout(&mut self, timeout: Option<u64>) {
        self.read_deadline = timeout.map(|a| ::io::timer::now() + a).unwrap_or(0);
    }
    fn set_write_timeout(&mut self, timeout: Option<u64>) {
        self.write_deadline = timeout.map(|a| ::io::timer::now() + a).unwrap_or(0);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Timeout helpers
//
// The read/write functions below are the helpers for reading/writing a socket
// with a possible deadline specified. This is generally viewed as a timed out
// I/O operation.
//
// From the application's perspective, timeouts apply to the I/O object, not to
// the underlying file descriptor (it's one timeout per object). This means that
// we can't use the SO_RCVTIMEO and corresponding send timeout option.
//
// The next idea to implement timeouts would be to use nonblocking I/O. An
// invocation of select() would wait (with a timeout) for a socket to be ready.
// Once its ready, we can perform the operation. Note that the operation *must*
// be nonblocking, even though select() says the socket is ready. This is
// because some other thread could have come and stolen our data (handles can be
// cloned).
//
// To implement nonblocking I/O, the first option we have is to use the
// O_NONBLOCK flag. Remember though that this is a global setting, affecting all
// I/O objects, so this was initially viewed as unwise.
//
// It turns out that there's this nifty MSG_DONTWAIT flag which can be passed to
// send/recv, but the niftiness wears off once you realize it only works well on
// linux [1] [2]. This means that it's pretty easy to get a nonblocking
// operation on linux (no flag fiddling, no affecting other objects), but not on
// other platforms.
//
// To work around this constraint on other platforms, we end up using the
// original strategy of flipping the O_NONBLOCK flag. As mentioned before, this
// could cause other objects' blocking operations to suddenly become
// nonblocking. To get around this, a "blocking operation" which returns EAGAIN
// falls back to using the same code path as nonblocking operations, but with an
// infinite timeout (select + send/recv). This helps emulate blocking
// reads/writes despite the underlying descriptor being nonblocking, as well as
// optimizing the fast path of just hitting one syscall in the good case.
//
// As a final caveat, this implementation uses a mutex so only one thread is
// doing a nonblocking operation at at time. This is the operation that comes
// after the select() (at which point we think the socket is ready). This is
// done for sanity to ensure that the state of the O_NONBLOCK flag is what we
// expect (wouldn't want someone turning it on when it should be off!). All
// operations performed in the lock are *nonblocking* to avoid holding the mutex
// forever.
//
// So, in summary, linux uses MSG_DONTWAIT and doesn't need mutexes, everyone
// else uses O_NONBLOCK and mutexes with some trickery to make sure blocking
// reads/writes are still blocking.
//
// Fun, fun!
//
// [1] http://twistedmatrix.com/pipermail/twisted-commits/2012-April/034692.html
// [2] http://stackoverflow.com/questions/19819198/does-send-msg-dontwait

pub fn read<T>(fd: sock_t,
               deadline: u64,
               lock: || -> T,
               read: |bool| -> libc::c_int) -> IoResult<uint> {
    let mut ret = -1;
    if deadline == 0 {
        ret = retry(|| read(false));
    }

    if deadline != 0 || (ret == -1 && util::wouldblock()) {
        let deadline = match deadline {
            0 => None,
            n => Some(n),
        };
        loop {
            // With a timeout, first we wait for the socket to become
            // readable using select(), specifying the relevant timeout for
            // our previously set deadline.
            try!(util::await(fd, deadline, util::Readable));

            // At this point, we're still within the timeout, and we've
            // determined that the socket is readable (as returned by
            // select). We must still read the socket in *nonblocking* mode
            // because some other thread could come steal our data. If we
            // fail to read some data, we retry (hence the outer loop) and
            // wait for the socket to become readable again.
            let _guard = lock();
            match retry(|| read(deadline.is_some())) {
                -1 if util::wouldblock() => { assert!(deadline.is_some()); }
                -1 => return Err(last_error()),
               n => { ret = n; break }
            }
        }
    }

    match ret {
        0 => Err(util::eof()),
        n if n < 0 => Err(last_error()),
        n => Ok(n as uint)
    }
}

pub fn write<T>(fd: sock_t,
                deadline: u64,
                buf: &[u8],
                write_everything: bool,
                lock: || -> T,
                write: |bool, *const u8, uint| -> i64) -> IoResult<uint> {
    let mut ret = -1;
    let mut written = 0;
    if deadline == 0 {
        if write_everything {
            ret = keep_going(buf, |inner, len| {
                written = buf.len() - len;
                write(false, inner, len)
            });
        } else {
            ret = retry(|| {
                write(false, buf.as_ptr(), buf.len()) as libc::c_int
            }) as i64;
            if ret > 0 { written = ret as uint; }
        }
    }

    if deadline != 0 || (ret == -1 && util::wouldblock()) {
        let deadline = match deadline {
            0 => None,
            n => Some(n),
        };
        while written < buf.len() && (write_everything || written == 0) {
            // As with read(), first wait for the socket to be ready for
            // the I/O operation.
            match util::await(fd, deadline, util::Writable) {
                Err(ref e) if e.code == libc::EOF as uint && written > 0 => {
                    assert!(deadline.is_some());
                    return Err(util::short_write(written, "short write"))
                }
                Err(e) => return Err(e),
                Ok(()) => {}
            }

            // Also as with read(), we use MSG_DONTWAIT to guard ourselves
            // against unforeseen circumstances.
            let _guard = lock();
            let ptr = buf.slice_from(written).as_ptr();
            let len = buf.len() - written;
            match retry(|| write(deadline.is_some(), ptr, len) as libc::c_int) {
                -1 if util::wouldblock() => {}
                -1 => return Err(last_error()),
                n => { written += n as uint; }
            }
        }
        ret = 0;
    }
    if ret < 0 {
        Err(last_error())
    } else {
        Ok(written)
    }
}
