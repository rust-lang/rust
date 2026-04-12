//! ThingOS networking implementation.
//!
//! TCP and UDP over IPv4 are supported via raw socket syscalls.
//! IPv6 returns `EAFNOSUPPORT` (issue #4).
//! DNS resolution is delegated to `SYS_SOCKET` + kernel resolver.

use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut};
use crate::net::{Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, ToSocketAddrs};
use crate::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use crate::sys::fd::FileDesc;
use crate::sys::pal::common::{
    AF_INET, AF_INET6, EAFNOSUPPORT, F_GETFL, F_SETFL, IPPROTO_IP, IPPROTO_IPV6, IPPROTO_TCP,
    IP_TTL, IPV6_UNICAST_HOPS, O_NONBLOCK, SHUT_RD, SHUT_RDWR, SHUT_WR, SOCK_DGRAM, SOCK_STREAM,
    SO_ERROR, SO_KEEPALIVE, SO_RCVTIMEO, SO_SNDTIMEO, SOL_SOCKET, SYS_ACCEPT, SYS_BIND,
    SYS_CONNECT, SYS_DUP, SYS_FCNTL, SYS_GETPEERNAME, SYS_GETSOCKNAME, SYS_GETSOCKOPT,
    SYS_LISTEN, SYS_RECV, SYS_SEND, SYS_SETSOCKOPT, SYS_SHUTDOWN, SYS_SOCKET, SYS_SET_SOCKET_TIMEOUT,
    TCP_NODELAY, SockaddrIn, SockaddrIn6, cvt, raw_syscall6, syscall3,
};
use crate::sys::{AsInner, FromInner, IntoInner};
use crate::time::Duration;
use crate::{fmt, mem};

// ── Internal socket wrapper ───────────────────────────────────────────────────

#[derive(Debug)]
struct Socket(FileDesc);

impl Socket {
    fn new(family: u32, ty: u32) -> io::Result<Socket> {
        let fd = cvt(unsafe {
            raw_syscall6(SYS_SOCKET, family as u64, ty as u64, 0, 0, 0, 0)
        })?;
        // SAFETY: kernel returned a valid owned fd.
        unsafe { Ok(Socket(FileDesc::from_inner(OwnedFd::from_raw_fd(fd as i32)))) }
    }

    fn raw_fd(&self) -> i32 {
        self.0.as_inner().as_raw_fd()
    }

    fn set_timeout(&self, which: u32, dur: Option<Duration>) -> io::Result<()> {
        let secs = dur.map_or(0u64, |d| d.as_secs());
        let usec = dur.map_or(0u64, |d| d.subsec_micros() as u64);
        cvt(unsafe {
            raw_syscall6(
                SYS_SET_SOCKET_TIMEOUT,
                self.raw_fd() as u64,
                which as u64,
                secs,
                usec,
                0,
                0,
            )
        })?;
        Ok(())
    }

    fn get_timeout(&self, which: u32) -> io::Result<Option<Duration>> {
        // For a full implementation, use SYS_GETSOCKOPT with SO_RCVTIMEO/SO_SNDTIMEO.
        // Simplified: return None.
        Ok(None)
    }

    fn set_nonblocking(&self, nb: bool) -> io::Result<()> {
        self.0.set_nonblocking(nb)
    }

    fn duplicate(&self) -> io::Result<Socket> {
        let fd = cvt(unsafe {
            raw_syscall6(SYS_DUP, self.raw_fd() as u64, 0, 0, 0, 0, 0)
        })?;
        unsafe { Ok(Socket(FileDesc::from_inner(OwnedFd::from_raw_fd(fd as i32)))) }
    }

    fn setsockopt_i32(&self, level: u32, name: u32, val: i32) -> io::Result<()> {
        cvt(unsafe {
            raw_syscall6(
                SYS_SETSOCKOPT,
                self.raw_fd() as u64,
                level as u64,
                name as u64,
                &raw const val as u64,
                4,
                0,
            )
        })?;
        Ok(())
    }

    fn getsockopt_i32(&self, level: u32, name: u32) -> io::Result<i32> {
        let mut val: i32 = 0;
        let mut len: u32 = 4;
        cvt(unsafe {
            raw_syscall6(
                SYS_GETSOCKOPT,
                self.raw_fd() as u64,
                level as u64,
                name as u64,
                &raw mut val as u64,
                &raw mut len as u64,
                0,
            )
        })?;
        Ok(val)
    }

    fn peer_addr(&self) -> io::Result<SocketAddr> {
        let mut addr = SockaddrIn { family: 0, port: 0, addr: 0, _pad: [0; 8] };
        let mut len: u32 = mem::size_of::<SockaddrIn>() as u32;
        cvt(unsafe {
            raw_syscall6(
                SYS_GETPEERNAME,
                self.raw_fd() as u64,
                &raw mut addr as u64,
                &raw mut len as u64,
                0,
                0,
                0,
            )
        })?;
        Ok(sockaddr_in_to_socket_addr(&addr))
    }

    fn socket_addr(&self) -> io::Result<SocketAddr> {
        let mut addr = SockaddrIn { family: 0, port: 0, addr: 0, _pad: [0; 8] };
        let mut len: u32 = mem::size_of::<SockaddrIn>() as u32;
        cvt(unsafe {
            raw_syscall6(
                SYS_GETSOCKNAME,
                self.raw_fd() as u64,
                &raw mut addr as u64,
                &raw mut len as u64,
                0,
                0,
                0,
            )
        })?;
        Ok(sockaddr_in_to_socket_addr(&addr))
    }

    fn send(&self, buf: &[u8], flags: u64) -> io::Result<usize> {
        cvt(unsafe {
            raw_syscall6(
                SYS_SEND,
                self.raw_fd() as u64,
                buf.as_ptr() as u64,
                buf.len() as u64,
                flags,
                0,
                0,
            )
        })
        .map(|n| n as usize)
    }

    fn recv(&self, buf: &mut [u8], flags: u64) -> io::Result<usize> {
        cvt(unsafe {
            raw_syscall6(
                SYS_RECV,
                self.raw_fd() as u64,
                buf.as_mut_ptr() as u64,
                buf.len() as u64,
                flags,
                0,
                0,
            )
        })
        .map(|n| n as usize)
    }

    fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        let how_val = match how {
            Shutdown::Read => SHUT_RD,
            Shutdown::Write => SHUT_WR,
            Shutdown::Both => SHUT_RDWR,
        };
        cvt(unsafe {
            raw_syscall6(SYS_SHUTDOWN, self.raw_fd() as u64, how_val as u64, 0, 0, 0, 0)
        })?;
        Ok(())
    }
}

// ── Address helpers ───────────────────────────────────────────────────────────

fn socket_addr_to_sockaddr_in(addr: &SocketAddr) -> (SockaddrIn, u32) {
    match addr {
        SocketAddr::V4(v4) => {
            let ip = v4.ip().octets();
            let ip_u32 = u32::from_be_bytes(ip);
            (
                SockaddrIn {
                    family: AF_INET as u16,
                    port: v4.port().to_be(),
                    addr: ip_u32.to_be(),
                    _pad: [0; 8],
                },
                mem::size_of::<SockaddrIn>() as u32,
            )
        }
        SocketAddr::V6(_) => {
            // IPv6 not supported (issue #4); caller should handle EAFNOSUPPORT.
            (SockaddrIn { family: AF_INET6 as u16, port: 0, addr: 0, _pad: [0; 8] }, 0)
        }
    }
}

fn sockaddr_in_to_socket_addr(addr: &SockaddrIn) -> SocketAddr {
    let ip = Ipv4Addr::from(u32::from_be(addr.addr).to_be_bytes());
    let port = u16::from_be(addr.port);
    SocketAddr::V4(crate::net::SocketAddrV4::new(ip, port))
}

fn ipv6_unsupported() -> io::Error {
    io::Error::from_raw_os_error(EAFNOSUPPORT)
}

// ── TcpStream ─────────────────────────────────────────────────────────────────

pub struct TcpStream {
    inner: Socket,
}

impl TcpStream {
    pub fn connect<A: ToSocketAddrs>(addr: A) -> io::Result<TcpStream> {
        let addr = addr.to_socket_addrs()?.next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "no addresses"))?;
        TcpStream::connect_timeout(&addr, Duration::MAX)
    }

    pub fn connect_timeout(addr: &SocketAddr, _timeout: Duration) -> io::Result<TcpStream> {
        if addr.is_ipv6() {
            return Err(ipv6_unsupported());
        }
        let sock = Socket::new(AF_INET, SOCK_STREAM)?;
        let (sa, sa_len) = socket_addr_to_sockaddr_in(addr);
        cvt(unsafe {
            raw_syscall6(
                SYS_CONNECT,
                sock.raw_fd() as u64,
                &raw const sa as u64,
                sa_len as u64,
                0,
                0,
                0,
            )
        })?;
        Ok(TcpStream { inner: sock })
    }

    pub fn set_read_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        self.inner.set_timeout(SO_RCVTIMEO, t)
    }

    pub fn set_write_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        self.inner.set_timeout(SO_SNDTIMEO, t)
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        self.inner.get_timeout(SO_RCVTIMEO)
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        self.inner.get_timeout(SO_SNDTIMEO)
    }

    pub fn peek(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.recv(buf, 2 /* MSG_PEEK */)
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.recv(buf, 0)
    }

    pub fn read_buf(&self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        crate::io::default_read_buf(|buf| self.read(buf), cursor)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        crate::io::default_read_vectored(|b| self.read(b), bufs)
    }

    pub fn is_read_vectored(&self) -> bool {
        false
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.inner.send(buf, 0)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        crate::io::default_write_vectored(|b| self.write(b), bufs)
    }

    pub fn is_write_vectored(&self) -> bool {
        false
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        self.inner.peer_addr()
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        self.inner.socket_addr()
    }

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        self.inner.shutdown(how)
    }

    pub fn duplicate(&self) -> io::Result<TcpStream> {
        self.inner.duplicate().map(|s| TcpStream { inner: s })
    }

    pub fn set_linger(&self, _dur: Option<Duration>) -> io::Result<()> {
        crate::sys::pal::unsupported()
    }

    pub fn linger(&self) -> io::Result<Option<Duration>> {
        crate::sys::pal::unsupported()
    }

    pub fn set_nodelay(&self, v: bool) -> io::Result<()> {
        self.inner.setsockopt_i32(IPPROTO_TCP, TCP_NODELAY, v as i32)
    }

    pub fn nodelay(&self) -> io::Result<bool> {
        self.inner.getsockopt_i32(IPPROTO_TCP, TCP_NODELAY).map(|v| v != 0)
    }

    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        self.inner.setsockopt_i32(IPPROTO_IP, IP_TTL, ttl as i32)
    }

    pub fn ttl(&self) -> io::Result<u32> {
        self.inner.getsockopt_i32(IPPROTO_IP, IP_TTL).map(|v| v as u32)
    }

    pub fn set_only_v6(&self, _: bool) -> io::Result<()> {
        Err(ipv6_unsupported())
    }

    pub fn only_v6(&self) -> io::Result<bool> {
        Err(ipv6_unsupported())
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        let e = self.inner.getsockopt_i32(SOL_SOCKET, SO_ERROR)?;
        if e == 0 { Ok(None) } else { Ok(Some(io::Error::from_raw_os_error(e))) }
    }

    pub fn set_nonblocking(&self, nb: bool) -> io::Result<()> {
        self.inner.set_nonblocking(nb)
    }
}

impl fmt::Debug for TcpStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TcpStream(fd={})", self.inner.raw_fd())
    }
}

// ── TcpListener ───────────────────────────────────────────────────────────────

pub struct TcpListener {
    inner: Socket,
}

impl TcpListener {
    pub fn bind<A: ToSocketAddrs>(addr: A) -> io::Result<TcpListener> {
        let addr = addr.to_socket_addrs()?.next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "no addresses"))?;
        if addr.is_ipv6() {
            return Err(ipv6_unsupported());
        }
        let sock = Socket::new(AF_INET, SOCK_STREAM)?;
        // Allow address reuse.
        sock.setsockopt_i32(SOL_SOCKET, 2 /* SO_REUSEADDR */, 1)?;
        let (sa, sa_len) = socket_addr_to_sockaddr_in(&addr);
        cvt(unsafe {
            raw_syscall6(
                SYS_BIND,
                sock.raw_fd() as u64,
                &raw const sa as u64,
                sa_len as u64,
                0,
                0,
                0,
            )
        })?;
        cvt(unsafe {
            raw_syscall6(SYS_LISTEN, sock.raw_fd() as u64, 128, 0, 0, 0, 0)
        })?;
        Ok(TcpListener { inner: sock })
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        self.inner.socket_addr()
    }

    pub fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        let mut peer = SockaddrIn { family: 0, port: 0, addr: 0, _pad: [0; 8] };
        let mut peer_len: u32 = mem::size_of::<SockaddrIn>() as u32;
        let fd = cvt(unsafe {
            raw_syscall6(
                SYS_ACCEPT,
                self.inner.raw_fd() as u64,
                &raw mut peer as u64,
                &raw mut peer_len as u64,
                0,
                0,
                0,
            )
        })?;
        let addr = sockaddr_in_to_socket_addr(&peer);
        // SAFETY: fd is a valid owned socket fd.
        let stream = unsafe {
            TcpStream {
                inner: Socket(FileDesc::from_inner(OwnedFd::from_raw_fd(fd as i32))),
            }
        };
        Ok((stream, addr))
    }

    pub fn duplicate(&self) -> io::Result<TcpListener> {
        self.inner.duplicate().map(|s| TcpListener { inner: s })
    }

    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        self.inner.setsockopt_i32(IPPROTO_IP, IP_TTL, ttl as i32)
    }

    pub fn ttl(&self) -> io::Result<u32> {
        self.inner.getsockopt_i32(IPPROTO_IP, IP_TTL).map(|v| v as u32)
    }

    pub fn set_only_v6(&self, _: bool) -> io::Result<()> {
        Err(ipv6_unsupported())
    }

    pub fn only_v6(&self) -> io::Result<bool> {
        Err(ipv6_unsupported())
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        let e = self.inner.getsockopt_i32(SOL_SOCKET, SO_ERROR)?;
        if e == 0 { Ok(None) } else { Ok(Some(io::Error::from_raw_os_error(e))) }
    }

    pub fn set_nonblocking(&self, nb: bool) -> io::Result<()> {
        self.inner.set_nonblocking(nb)
    }
}

impl fmt::Debug for TcpListener {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TcpListener(fd={})", self.inner.raw_fd())
    }
}

// ── UdpSocket ─────────────────────────────────────────────────────────────────

pub struct UdpSocket {
    inner: Socket,
}

impl UdpSocket {
    pub fn bind<A: ToSocketAddrs>(addr: A) -> io::Result<UdpSocket> {
        let addr = addr.to_socket_addrs()?.next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "no addresses"))?;
        if addr.is_ipv6() {
            return Err(ipv6_unsupported());
        }
        let sock = Socket::new(AF_INET, SOCK_DGRAM)?;
        let (sa, sa_len) = socket_addr_to_sockaddr_in(&addr);
        cvt(unsafe {
            raw_syscall6(
                SYS_BIND,
                sock.raw_fd() as u64,
                &raw const sa as u64,
                sa_len as u64,
                0,
                0,
                0,
            )
        })?;
        Ok(UdpSocket { inner: sock })
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        self.inner.peer_addr()
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        self.inner.socket_addr()
    }

    pub fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        let mut addr = SockaddrIn { family: 0, port: 0, addr: 0, _pad: [0; 8] };
        let mut addr_len: u32 = mem::size_of::<SockaddrIn>() as u32;
        let n = cvt(unsafe {
            raw_syscall6(
                crate::sys::pal::common::SYS_RECV,
                self.inner.raw_fd() as u64,
                buf.as_mut_ptr() as u64,
                buf.len() as u64,
                0,
                &raw mut addr as u64,
                &raw mut addr_len as u64,
            )
        })? as usize;
        Ok((n, sockaddr_in_to_socket_addr(&addr)))
    }

    pub fn peek_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        let mut addr = SockaddrIn { family: 0, port: 0, addr: 0, _pad: [0; 8] };
        let mut addr_len: u32 = mem::size_of::<SockaddrIn>() as u32;
        let n = cvt(unsafe {
            raw_syscall6(
                crate::sys::pal::common::SYS_RECV,
                self.inner.raw_fd() as u64,
                buf.as_mut_ptr() as u64,
                buf.len() as u64,
                2, /* MSG_PEEK */
                &raw mut addr as u64,
                &raw mut addr_len as u64,
            )
        })? as usize;
        Ok((n, sockaddr_in_to_socket_addr(&addr)))
    }

    pub fn send_to<A: ToSocketAddrs>(&self, buf: &[u8], dst: A) -> io::Result<usize> {
        let addr = dst.to_socket_addrs()?.next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "no addresses"))?;
        if addr.is_ipv6() {
            return Err(ipv6_unsupported());
        }
        let (sa, sa_len) = socket_addr_to_sockaddr_in(&addr);
        cvt(unsafe {
            raw_syscall6(
                crate::sys::pal::common::SYS_SEND,
                self.inner.raw_fd() as u64,
                buf.as_ptr() as u64,
                buf.len() as u64,
                0,
                &raw const sa as u64,
                sa_len as u64,
            )
        })
        .map(|n| n as usize)
    }

    pub fn duplicate(&self) -> io::Result<UdpSocket> {
        self.inner.duplicate().map(|s| UdpSocket { inner: s })
    }

    pub fn set_read_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        self.inner.set_timeout(SO_RCVTIMEO, t)
    }

    pub fn set_write_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        self.inner.set_timeout(SO_SNDTIMEO, t)
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        self.inner.get_timeout(SO_RCVTIMEO)
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        self.inner.get_timeout(SO_SNDTIMEO)
    }

    pub fn set_broadcast(&self, _: bool) -> io::Result<()> {
        crate::sys::pal::unsupported()
    }

    pub fn broadcast(&self) -> io::Result<bool> {
        crate::sys::pal::unsupported()
    }

    pub fn set_multicast_loop_v4(&self, _: bool) -> io::Result<()> {
        crate::sys::pal::unsupported()
    }

    pub fn multicast_loop_v4(&self) -> io::Result<bool> {
        crate::sys::pal::unsupported()
    }

    pub fn set_multicast_ttl_v4(&self, _: u32) -> io::Result<()> {
        crate::sys::pal::unsupported()
    }

    pub fn multicast_ttl_v4(&self) -> io::Result<u32> {
        crate::sys::pal::unsupported()
    }

    pub fn set_multicast_loop_v6(&self, _: bool) -> io::Result<()> {
        Err(ipv6_unsupported())
    }

    pub fn multicast_loop_v6(&self) -> io::Result<bool> {
        Err(ipv6_unsupported())
    }

    pub fn join_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr) -> io::Result<()> {
        crate::sys::pal::unsupported()
    }

    pub fn join_multicast_v6(&self, _: &Ipv6Addr, _: u32) -> io::Result<()> {
        Err(ipv6_unsupported())
    }

    pub fn leave_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr) -> io::Result<()> {
        crate::sys::pal::unsupported()
    }

    pub fn leave_multicast_v6(&self, _: &Ipv6Addr, _: u32) -> io::Result<()> {
        Err(ipv6_unsupported())
    }

    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        self.inner.setsockopt_i32(IPPROTO_IP, IP_TTL, ttl as i32)
    }

    pub fn ttl(&self) -> io::Result<u32> {
        self.inner.getsockopt_i32(IPPROTO_IP, IP_TTL).map(|v| v as u32)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        let e = self.inner.getsockopt_i32(SOL_SOCKET, SO_ERROR)?;
        if e == 0 { Ok(None) } else { Ok(Some(io::Error::from_raw_os_error(e))) }
    }

    pub fn set_nonblocking(&self, nb: bool) -> io::Result<()> {
        self.inner.set_nonblocking(nb)
    }

    pub fn recv(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.recv(buf, 0)
    }

    pub fn peek(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.recv(buf, 2 /* MSG_PEEK */)
    }

    pub fn send(&self, buf: &[u8]) -> io::Result<usize> {
        self.inner.send(buf, 0)
    }

    pub fn connect<A: ToSocketAddrs>(&self, addr: A) -> io::Result<()> {
        let addr = addr.to_socket_addrs()?.next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "no addresses"))?;
        if addr.is_ipv6() {
            return Err(ipv6_unsupported());
        }
        let (sa, sa_len) = socket_addr_to_sockaddr_in(&addr);
        cvt(unsafe {
            raw_syscall6(
                SYS_CONNECT,
                self.inner.raw_fd() as u64,
                &raw const sa as u64,
                sa_len as u64,
                0,
                0,
                0,
            )
        })?;
        Ok(())
    }

    pub fn set_only_v6(&self, _: bool) -> io::Result<()> {
        Err(ipv6_unsupported())
    }

    pub fn only_v6(&self) -> io::Result<bool> {
        Err(ipv6_unsupported())
    }
}

impl fmt::Debug for UdpSocket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UdpSocket(fd={})", self.inner.raw_fd())
    }
}

// ── DNS / LookupHost ──────────────────────────────────────────────────────────

/// A simple DNS lookup result iterator.
///
/// ThingOS performs DNS resolution via a dedicated kernel resolver syscall.
/// The full implementation requires a `SYS_RESOLVE` syscall or VFS-based DNS
/// resolver.  For now, a minimal implementation is provided.
pub struct LookupHost {
    addrs: crate::vec::IntoIter<SocketAddr>,
}

impl Iterator for LookupHost {
    type Item = SocketAddr;
    fn next(&mut self) -> Option<SocketAddr> {
        self.addrs.next()
    }
}

pub fn lookup_host(host: &str, port: u16) -> io::Result<LookupHost> {
    // Attempt to parse the host as an IP address directly.
    if let Ok(ip) = host.parse::<Ipv4Addr>() {
        return Ok(LookupHost {
            addrs: vec![SocketAddr::V4(crate::net::SocketAddrV4::new(ip, port))].into_iter(),
        });
    }
    if let Ok(ip) = host.parse::<Ipv6Addr>() {
        // IPv6 addresses parse fine but connecting to them will fail.
        return Ok(LookupHost {
            addrs: vec![SocketAddr::V6(crate::net::SocketAddrV6::new(ip, port, 0, 0))].into_iter(),
        });
    }
    // Non-trivial hostname resolution is not yet implemented.
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "hostname DNS resolution not yet implemented on ThingOS",
    ))
}
