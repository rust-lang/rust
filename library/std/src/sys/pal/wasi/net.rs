#![forbid(unsafe_op_in_unsafe_fn)]

use super::err2io;
use super::fd::WasiFd;
use crate::fmt;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut};
use crate::net::{Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr};
use crate::os::wasi::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, RawFd};
use crate::sys::unsupported;
use crate::sys_common::{AsInner, FromInner, IntoInner};
use crate::time::Duration;

pub struct Socket(WasiFd);

pub struct TcpStream {
    inner: Socket,
}

impl AsInner<WasiFd> for Socket {
    #[inline]
    fn as_inner(&self) -> &WasiFd {
        &self.0
    }
}

impl IntoInner<WasiFd> for Socket {
    fn into_inner(self) -> WasiFd {
        self.0
    }
}

impl FromInner<WasiFd> for Socket {
    fn from_inner(inner: WasiFd) -> Socket {
        Socket(inner)
    }
}

impl AsFd for Socket {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

impl AsRawFd for Socket {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.0.as_raw_fd()
    }
}

impl IntoRawFd for Socket {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_raw_fd()
    }
}

impl FromRawFd for Socket {
    unsafe fn from_raw_fd(raw_fd: RawFd) -> Self {
        unsafe { Self(FromRawFd::from_raw_fd(raw_fd)) }
    }
}

impl TcpStream {
    pub fn connect(_: io::Result<&SocketAddr>) -> io::Result<TcpStream> {
        unsupported()
    }

    pub fn connect_timeout(_: &SocketAddr, _: Duration) -> io::Result<TcpStream> {
        unsupported()
    }

    pub fn set_read_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        unsupported()
    }

    pub fn set_write_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        unsupported()
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        unsupported()
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        unsupported()
    }

    pub fn peek(&self, _: &mut [u8]) -> io::Result<usize> {
        unsupported()
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.read_vectored(&mut [IoSliceMut::new(buf)])
    }

    pub fn read_buf(&self, buf: BorrowedCursor<'_>) -> io::Result<()> {
        self.socket().as_inner().read_buf(buf)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.socket().as_inner().read(bufs)
    }

    pub fn is_read_vectored(&self) -> bool {
        true
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.write_vectored(&[IoSlice::new(buf)])
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.socket().as_inner().write(bufs)
    }

    pub fn is_write_vectored(&self) -> bool {
        true
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        unsupported()
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        unsupported()
    }

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        let wasi_how = match how {
            Shutdown::Read => wasi::SDFLAGS_RD,
            Shutdown::Write => wasi::SDFLAGS_WR,
            Shutdown::Both => wasi::SDFLAGS_RD | wasi::SDFLAGS_WR,
        };

        unsafe { wasi::sock_shutdown(self.socket().as_raw_fd() as _, wasi_how).map_err(err2io) }
    }

    pub fn duplicate(&self) -> io::Result<TcpStream> {
        unsupported()
    }

    pub fn set_linger(&self, _: Option<Duration>) -> io::Result<()> {
        unsupported()
    }

    pub fn linger(&self) -> io::Result<Option<Duration>> {
        unsupported()
    }

    pub fn set_nodelay(&self, _: bool) -> io::Result<()> {
        unsupported()
    }

    pub fn nodelay(&self) -> io::Result<bool> {
        unsupported()
    }

    pub fn set_ttl(&self, _: u32) -> io::Result<()> {
        unsupported()
    }

    pub fn ttl(&self) -> io::Result<u32> {
        unsupported()
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        unsupported()
    }

    pub fn set_nonblocking(&self, state: bool) -> io::Result<()> {
        let fdstat = unsafe {
            wasi::fd_fdstat_get(self.socket().as_inner().as_raw_fd() as wasi::Fd).map_err(err2io)?
        };

        let mut flags = fdstat.fs_flags;

        if state {
            flags |= wasi::FDFLAGS_NONBLOCK;
        } else {
            flags &= !wasi::FDFLAGS_NONBLOCK;
        }

        unsafe {
            wasi::fd_fdstat_set_flags(self.socket().as_inner().as_raw_fd() as wasi::Fd, flags)
                .map_err(err2io)
        }
    }

    #[inline]
    pub fn socket(&self) -> &Socket {
        &self.inner
    }

    pub fn into_socket(self) -> Socket {
        self.inner
    }
}

impl FromInner<Socket> for TcpStream {
    fn from_inner(socket: Socket) -> TcpStream {
        TcpStream { inner: socket }
    }
}

impl fmt::Debug for TcpStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TcpStream").field("fd", &self.inner.as_raw_fd()).finish()
    }
}

pub struct TcpListener {
    inner: Socket,
}

impl TcpListener {
    pub fn bind(_: io::Result<&SocketAddr>) -> io::Result<TcpListener> {
        unsupported()
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        unsupported()
    }

    pub fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        let fd = unsafe {
            wasi::sock_accept(self.as_inner().as_inner().as_raw_fd() as _, 0).map_err(err2io)?
        };

        Ok((
            TcpStream::from_inner(unsafe { Socket::from_raw_fd(fd as _) }),
            // WASI has no concept of SocketAddr yet
            // return an unspecified IPv4Addr
            SocketAddr::new(Ipv4Addr::UNSPECIFIED.into(), 0),
        ))
    }

    pub fn duplicate(&self) -> io::Result<TcpListener> {
        unsupported()
    }

    pub fn set_ttl(&self, _: u32) -> io::Result<()> {
        unsupported()
    }

    pub fn ttl(&self) -> io::Result<u32> {
        unsupported()
    }

    pub fn set_only_v6(&self, _: bool) -> io::Result<()> {
        unsupported()
    }

    pub fn only_v6(&self) -> io::Result<bool> {
        unsupported()
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        unsupported()
    }

    pub fn set_nonblocking(&self, state: bool) -> io::Result<()> {
        let fdstat = unsafe {
            wasi::fd_fdstat_get(self.socket().as_inner().as_raw_fd() as wasi::Fd).map_err(err2io)?
        };

        let mut flags = fdstat.fs_flags;

        if state {
            flags |= wasi::FDFLAGS_NONBLOCK;
        } else {
            flags &= !wasi::FDFLAGS_NONBLOCK;
        }

        unsafe {
            wasi::fd_fdstat_set_flags(self.socket().as_inner().as_raw_fd() as wasi::Fd, flags)
                .map_err(err2io)
        }
    }

    #[inline]
    pub fn socket(&self) -> &Socket {
        &self.inner
    }

    pub fn into_socket(self) -> Socket {
        self.inner
    }
}

impl AsInner<Socket> for TcpListener {
    #[inline]
    fn as_inner(&self) -> &Socket {
        &self.inner
    }
}

impl IntoInner<Socket> for TcpListener {
    fn into_inner(self) -> Socket {
        self.inner
    }
}

impl FromInner<Socket> for TcpListener {
    fn from_inner(inner: Socket) -> TcpListener {
        TcpListener { inner }
    }
}

impl fmt::Debug for TcpListener {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TcpListener").field("fd", &self.inner.as_raw_fd()).finish()
    }
}

pub struct UdpSocket {
    inner: Socket,
}

impl UdpSocket {
    pub fn bind(_: io::Result<&SocketAddr>) -> io::Result<UdpSocket> {
        unsupported()
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        unsupported()
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        unsupported()
    }

    pub fn recv_from(&self, _: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        unsupported()
    }

    pub fn peek_from(&self, _: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        unsupported()
    }

    pub fn send_to(&self, _: &[u8], _: &SocketAddr) -> io::Result<usize> {
        unsupported()
    }

    pub fn duplicate(&self) -> io::Result<UdpSocket> {
        unsupported()
    }

    pub fn set_read_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        unsupported()
    }

    pub fn set_write_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        unsupported()
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        unsupported()
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        unsupported()
    }

    pub fn set_broadcast(&self, _: bool) -> io::Result<()> {
        unsupported()
    }

    pub fn broadcast(&self) -> io::Result<bool> {
        unsupported()
    }

    pub fn set_multicast_loop_v4(&self, _: bool) -> io::Result<()> {
        unsupported()
    }

    pub fn multicast_loop_v4(&self) -> io::Result<bool> {
        unsupported()
    }

    pub fn set_multicast_ttl_v4(&self, _: u32) -> io::Result<()> {
        unsupported()
    }

    pub fn multicast_ttl_v4(&self) -> io::Result<u32> {
        unsupported()
    }

    pub fn set_multicast_loop_v6(&self, _: bool) -> io::Result<()> {
        unsupported()
    }

    pub fn multicast_loop_v6(&self) -> io::Result<bool> {
        unsupported()
    }

    pub fn join_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr) -> io::Result<()> {
        unsupported()
    }

    pub fn join_multicast_v6(&self, _: &Ipv6Addr, _: u32) -> io::Result<()> {
        unsupported()
    }

    pub fn leave_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr) -> io::Result<()> {
        unsupported()
    }

    pub fn leave_multicast_v6(&self, _: &Ipv6Addr, _: u32) -> io::Result<()> {
        unsupported()
    }

    pub fn set_ttl(&self, _: u32) -> io::Result<()> {
        unsupported()
    }

    pub fn ttl(&self) -> io::Result<u32> {
        unsupported()
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        unsupported()
    }

    pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
        unsupported()
    }

    pub fn recv(&self, _: &mut [u8]) -> io::Result<usize> {
        unsupported()
    }

    pub fn peek(&self, _: &mut [u8]) -> io::Result<usize> {
        unsupported()
    }

    pub fn send(&self, _: &[u8]) -> io::Result<usize> {
        unsupported()
    }

    pub fn connect(&self, _: io::Result<&SocketAddr>) -> io::Result<()> {
        unsupported()
    }

    #[inline]
    pub fn socket(&self) -> &Socket {
        &self.inner
    }

    pub fn into_socket(self) -> Socket {
        self.inner
    }
}

impl AsInner<Socket> for UdpSocket {
    #[inline]
    fn as_inner(&self) -> &Socket {
        &self.inner
    }
}

impl IntoInner<Socket> for UdpSocket {
    fn into_inner(self) -> Socket {
        self.inner
    }
}

impl FromInner<Socket> for UdpSocket {
    fn from_inner(inner: Socket) -> UdpSocket {
        UdpSocket { inner }
    }
}

impl fmt::Debug for UdpSocket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UdpSocket").field("fd", &self.inner.as_raw_fd()).finish()
    }
}

pub struct LookupHost(!);

impl LookupHost {
    pub fn port(&self) -> u16 {
        self.0
    }
}

impl Iterator for LookupHost {
    type Item = SocketAddr;
    fn next(&mut self) -> Option<SocketAddr> {
        self.0
    }
}

impl<'a> TryFrom<&'a str> for LookupHost {
    type Error = io::Error;

    fn try_from(_v: &'a str) -> io::Result<LookupHost> {
        unsupported()
    }
}

impl<'a> TryFrom<(&'a str, u16)> for LookupHost {
    type Error = io::Error;

    fn try_from(_v: (&'a str, u16)) -> io::Result<LookupHost> {
        unsupported()
    }
}

#[allow(nonstandard_style)]
pub mod netc {
    pub const AF_INET: u8 = 0;
    pub const AF_INET6: u8 = 1;
    pub type sa_family_t = u8;

    #[derive(Copy, Clone)]
    pub struct in_addr {
        pub s_addr: u32,
    }

    #[derive(Copy, Clone)]
    pub struct sockaddr_in {
        #[allow(dead_code)]
        pub sin_family: sa_family_t,
        pub sin_port: u16,
        pub sin_addr: in_addr,
    }

    #[derive(Copy, Clone)]
    pub struct in6_addr {
        pub s6_addr: [u8; 16],
    }

    #[derive(Copy, Clone)]
    pub struct sockaddr_in6 {
        #[allow(dead_code)]
        pub sin6_family: sa_family_t,
        pub sin6_port: u16,
        pub sin6_addr: in6_addr,
        pub sin6_flowinfo: u32,
        pub sin6_scope_id: u32,
    }
}
