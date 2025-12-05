pub use moto_rt::netc;

use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut};
use crate::net::SocketAddr::{V4, V6};
use crate::net::{Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, ToSocketAddrs};
use crate::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, RawFd};
use crate::sys::fd::FileDesc;
use crate::sys::map_motor_error;
use crate::sys_common::{AsInner, FromInner, IntoInner};
use crate::time::Duration;

// We want to re-use as much of Rust's stdlib code as possible,
// and most of it is unixy, but with a lot of nesting.
#[derive(Debug)]
pub struct Socket(FileDesc);

#[derive(Debug)]
pub struct TcpStream {
    inner: Socket,
}

impl TcpStream {
    pub fn socket(&self) -> &Socket {
        &self.inner
    }

    pub fn into_socket(self) -> Socket {
        self.inner
    }

    pub fn connect<A: ToSocketAddrs>(addr: A) -> io::Result<TcpStream> {
        let addr = into_netc(&addr.to_socket_addrs()?.next().unwrap());
        moto_rt::net::tcp_connect(&addr, Duration::MAX, false)
            .map(|fd| Self { inner: unsafe { Socket::from_raw_fd(fd) } })
            .map_err(map_motor_error)
    }

    pub fn connect_timeout(addr: &SocketAddr, timeout: Duration) -> io::Result<TcpStream> {
        let addr = into_netc(addr);
        moto_rt::net::tcp_connect(&addr, timeout, false)
            .map(|fd| Self { inner: unsafe { Socket::from_raw_fd(fd) } })
            .map_err(map_motor_error)
    }

    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        moto_rt::net::set_read_timeout(self.inner.as_raw_fd(), timeout).map_err(map_motor_error)
    }

    pub fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        moto_rt::net::set_write_timeout(self.inner.as_raw_fd(), timeout).map_err(map_motor_error)
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        moto_rt::net::read_timeout(self.inner.as_raw_fd()).map_err(map_motor_error)
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        moto_rt::net::write_timeout(self.inner.as_raw_fd()).map_err(map_motor_error)
    }

    pub fn peek(&self, buf: &mut [u8]) -> io::Result<usize> {
        moto_rt::net::peek(self.inner.as_raw_fd(), buf).map_err(map_motor_error)
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        moto_rt::fs::read(self.inner.as_raw_fd(), buf).map_err(map_motor_error)
    }

    pub fn read_buf(&self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        crate::io::default_read_buf(|buf| self.read(buf), cursor)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let bufs: &mut [&mut [u8]] = unsafe { core::mem::transmute(bufs) };
        moto_rt::fs::read_vectored(self.inner.as_raw_fd(), bufs).map_err(map_motor_error)
    }

    pub fn is_read_vectored(&self) -> bool {
        true
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        moto_rt::fs::write(self.inner.as_raw_fd(), buf).map_err(map_motor_error)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let bufs: &[&[u8]] = unsafe { core::mem::transmute(bufs) };
        moto_rt::fs::write_vectored(self.inner.as_raw_fd(), bufs).map_err(map_motor_error)
    }

    pub fn is_write_vectored(&self) -> bool {
        true
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        moto_rt::net::peer_addr(self.inner.as_raw_fd())
            .map(|addr| from_netc(&addr))
            .map_err(map_motor_error)
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        moto_rt::net::socket_addr(self.inner.as_raw_fd())
            .map(|addr| from_netc(&addr))
            .map_err(map_motor_error)
    }

    pub fn shutdown(&self, shutdown: Shutdown) -> io::Result<()> {
        let shutdown = match shutdown {
            Shutdown::Read => moto_rt::net::SHUTDOWN_READ,
            Shutdown::Write => moto_rt::net::SHUTDOWN_WRITE,
            Shutdown::Both => moto_rt::net::SHUTDOWN_READ | moto_rt::net::SHUTDOWN_WRITE,
        };

        moto_rt::net::shutdown(self.inner.as_raw_fd(), shutdown).map_err(map_motor_error)
    }

    pub fn duplicate(&self) -> io::Result<TcpStream> {
        moto_rt::fs::duplicate(self.inner.as_raw_fd())
            .map(|fd| Self { inner: unsafe { Socket::from_raw_fd(fd) } })
            .map_err(map_motor_error)
    }

    pub fn set_linger(&self, timeout: Option<Duration>) -> io::Result<()> {
        moto_rt::net::set_linger(self.inner.as_raw_fd(), timeout).map_err(map_motor_error)
    }

    pub fn linger(&self) -> io::Result<Option<Duration>> {
        moto_rt::net::linger(self.inner.as_raw_fd()).map_err(map_motor_error)
    }

    pub fn set_nodelay(&self, nodelay: bool) -> io::Result<()> {
        moto_rt::net::set_nodelay(self.inner.as_raw_fd(), nodelay).map_err(map_motor_error)
    }

    pub fn nodelay(&self) -> io::Result<bool> {
        moto_rt::net::nodelay(self.inner.as_raw_fd()).map_err(map_motor_error)
    }

    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        moto_rt::net::set_ttl(self.inner.as_raw_fd(), ttl).map_err(map_motor_error)
    }

    pub fn ttl(&self) -> io::Result<u32> {
        moto_rt::net::ttl(self.inner.as_raw_fd()).map_err(map_motor_error)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        let e = moto_rt::net::take_error(self.inner.as_raw_fd()).map_err(map_motor_error)?;
        if e == moto_rt::E_OK { Ok(None) } else { Ok(Some(map_motor_error(e))) }
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        moto_rt::net::set_nonblocking(self.inner.as_raw_fd(), nonblocking).map_err(map_motor_error)
    }
}

#[derive(Debug)]
pub struct TcpListener {
    inner: Socket,
}

impl TcpListener {
    #[inline]
    pub fn socket(&self) -> &Socket {
        &self.inner
    }

    pub fn into_socket(self) -> Socket {
        self.inner
    }

    pub fn bind<A: ToSocketAddrs>(addr: A) -> io::Result<TcpListener> {
        let addr = into_netc(&addr.to_socket_addrs()?.next().unwrap());
        moto_rt::net::bind(moto_rt::net::PROTO_TCP, &addr)
            .map(|fd| Self { inner: unsafe { Socket::from_raw_fd(fd) } })
            .map_err(map_motor_error)
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        moto_rt::net::socket_addr(self.inner.as_raw_fd())
            .map(|addr| from_netc(&addr))
            .map_err(map_motor_error)
    }

    pub fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        moto_rt::net::accept(self.inner.as_raw_fd())
            .map(|(fd, addr)| {
                (TcpStream { inner: unsafe { Socket::from_raw_fd(fd) } }, from_netc(&addr))
            })
            .map_err(map_motor_error)
    }

    pub fn duplicate(&self) -> io::Result<TcpListener> {
        moto_rt::fs::duplicate(self.inner.as_raw_fd())
            .map(|fd| Self { inner: unsafe { Socket::from_raw_fd(fd) } })
            .map_err(map_motor_error)
    }

    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        moto_rt::net::set_ttl(self.inner.as_raw_fd(), ttl).map_err(map_motor_error)
    }

    pub fn ttl(&self) -> io::Result<u32> {
        moto_rt::net::ttl(self.inner.as_raw_fd()).map_err(map_motor_error)
    }

    pub fn set_only_v6(&self, only_v6: bool) -> io::Result<()> {
        moto_rt::net::set_only_v6(self.inner.as_raw_fd(), only_v6).map_err(map_motor_error)
    }

    pub fn only_v6(&self) -> io::Result<bool> {
        moto_rt::net::only_v6(self.inner.as_raw_fd()).map_err(map_motor_error)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        let e = moto_rt::net::take_error(self.inner.as_raw_fd()).map_err(map_motor_error)?;
        if e == moto_rt::E_OK { Ok(None) } else { Ok(Some(map_motor_error(e))) }
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        moto_rt::net::set_nonblocking(self.inner.as_raw_fd(), nonblocking).map_err(map_motor_error)
    }
}

#[derive(Debug)]
pub struct UdpSocket {
    inner: Socket,
}

impl UdpSocket {
    pub fn socket(&self) -> &Socket {
        &self.inner
    }

    pub fn into_socket(self) -> Socket {
        self.inner
    }

    pub fn bind<A: ToSocketAddrs>(addr: A) -> io::Result<UdpSocket> {
        let addr = into_netc(&addr.to_socket_addrs()?.next().unwrap());
        moto_rt::net::bind(moto_rt::net::PROTO_UDP, &addr)
            .map(|fd| Self { inner: unsafe { Socket::from_raw_fd(fd) } })
            .map_err(map_motor_error)
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        moto_rt::net::peer_addr(self.inner.as_raw_fd())
            .map(|addr| from_netc(&addr))
            .map_err(map_motor_error)
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        moto_rt::net::socket_addr(self.inner.as_raw_fd())
            .map(|addr| from_netc(&addr))
            .map_err(map_motor_error)
    }

    pub fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        moto_rt::net::udp_recv_from(self.inner.as_raw_fd(), buf)
            .map(|(sz, addr)| (sz, from_netc(&addr)))
            .map_err(map_motor_error)
    }

    pub fn peek_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        moto_rt::net::udp_peek_from(self.inner.as_raw_fd(), buf)
            .map(|(sz, addr)| (sz, from_netc(&addr)))
            .map_err(map_motor_error)
    }

    pub fn send_to(&self, buf: &[u8], addr: &SocketAddr) -> io::Result<usize> {
        let addr = into_netc(addr);
        moto_rt::net::udp_send_to(self.inner.as_raw_fd(), buf, &addr).map_err(map_motor_error)
    }

    pub fn duplicate(&self) -> io::Result<UdpSocket> {
        moto_rt::fs::duplicate(self.inner.as_raw_fd())
            .map(|fd| Self { inner: unsafe { Socket::from_raw_fd(fd) } })
            .map_err(map_motor_error)
    }

    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        moto_rt::net::set_read_timeout(self.inner.as_raw_fd(), timeout).map_err(map_motor_error)
    }

    pub fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        moto_rt::net::set_write_timeout(self.inner.as_raw_fd(), timeout).map_err(map_motor_error)
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        moto_rt::net::read_timeout(self.inner.as_raw_fd()).map_err(map_motor_error)
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        moto_rt::net::write_timeout(self.inner.as_raw_fd()).map_err(map_motor_error)
    }

    pub fn set_broadcast(&self, broadcast: bool) -> io::Result<()> {
        moto_rt::net::set_udp_broadcast(self.inner.as_raw_fd(), broadcast).map_err(map_motor_error)
    }

    pub fn broadcast(&self) -> io::Result<bool> {
        moto_rt::net::udp_broadcast(self.inner.as_raw_fd()).map_err(map_motor_error)
    }

    pub fn set_multicast_loop_v4(&self, val: bool) -> io::Result<()> {
        moto_rt::net::set_udp_multicast_loop_v4(self.inner.as_raw_fd(), val)
            .map_err(map_motor_error)
    }

    pub fn multicast_loop_v4(&self) -> io::Result<bool> {
        moto_rt::net::udp_multicast_loop_v4(self.inner.as_raw_fd()).map_err(map_motor_error)
    }

    pub fn set_multicast_ttl_v4(&self, val: u32) -> io::Result<()> {
        moto_rt::net::set_udp_multicast_ttl_v4(self.inner.as_raw_fd(), val).map_err(map_motor_error)
    }

    pub fn multicast_ttl_v4(&self) -> io::Result<u32> {
        moto_rt::net::udp_multicast_ttl_v4(self.inner.as_raw_fd()).map_err(map_motor_error)
    }

    pub fn set_multicast_loop_v6(&self, val: bool) -> io::Result<()> {
        moto_rt::net::set_udp_multicast_loop_v6(self.inner.as_raw_fd(), val)
            .map_err(map_motor_error)
    }

    pub fn multicast_loop_v6(&self) -> io::Result<bool> {
        moto_rt::net::udp_multicast_loop_v6(self.inner.as_raw_fd()).map_err(map_motor_error)
    }

    pub fn join_multicast_v4(&self, addr: &Ipv4Addr, iface: &Ipv4Addr) -> io::Result<()> {
        let addr = (*addr).into();
        let iface = (*iface).into();
        moto_rt::net::join_udp_multicast_v4(self.inner.as_raw_fd(), &addr, &iface)
            .map_err(map_motor_error)
    }

    pub fn join_multicast_v6(&self, addr: &Ipv6Addr, iface: u32) -> io::Result<()> {
        let addr = (*addr).into();
        moto_rt::net::join_udp_multicast_v6(self.inner.as_raw_fd(), &addr, iface)
            .map_err(map_motor_error)
    }

    pub fn leave_multicast_v4(&self, addr: &Ipv4Addr, iface: &Ipv4Addr) -> io::Result<()> {
        let addr = (*addr).into();
        let iface = (*iface).into();
        moto_rt::net::leave_udp_multicast_v4(self.inner.as_raw_fd(), &addr, &iface)
            .map_err(map_motor_error)
    }

    pub fn leave_multicast_v6(&self, addr: &Ipv6Addr, iface: u32) -> io::Result<()> {
        let addr = (*addr).into();
        moto_rt::net::leave_udp_multicast_v6(self.inner.as_raw_fd(), &addr, iface)
            .map_err(map_motor_error)
    }

    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        moto_rt::net::set_ttl(self.inner.as_raw_fd(), ttl).map_err(map_motor_error)
    }

    pub fn ttl(&self) -> io::Result<u32> {
        moto_rt::net::ttl(self.inner.as_raw_fd()).map_err(map_motor_error)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        moto_rt::net::take_error(self.inner.as_raw_fd())
            .map(|e| match e {
                moto_rt::E_OK => None,
                e => Some(map_motor_error(e)),
            })
            .map_err(map_motor_error)
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        moto_rt::net::set_nonblocking(self.inner.as_raw_fd(), nonblocking).map_err(map_motor_error)
    }

    pub fn recv(&self, buf: &mut [u8]) -> io::Result<usize> {
        moto_rt::fs::read(self.inner.as_raw_fd(), buf).map_err(map_motor_error)
    }

    pub fn peek(&self, buf: &mut [u8]) -> io::Result<usize> {
        moto_rt::net::peek(self.inner.as_raw_fd(), buf).map_err(map_motor_error)
    }

    pub fn send(&self, buf: &[u8]) -> io::Result<usize> {
        moto_rt::fs::write(self.inner.as_raw_fd(), buf).map_err(map_motor_error)
    }

    pub fn connect<A: ToSocketAddrs>(&self, addr: A) -> io::Result<()> {
        let addr = into_netc(&addr.to_socket_addrs()?.next().unwrap());
        moto_rt::net::udp_connect(self.inner.as_raw_fd(), &addr).map_err(map_motor_error)
    }
}

pub struct LookupHost {
    addresses: alloc::collections::VecDeque<netc::sockaddr>,
}

pub fn lookup_host(host: &str, port: u16) -> io::Result<LookupHost> {
    let (_port, addresses) = moto_rt::net::lookup_host(host, port).map_err(map_motor_error)?;
    Ok(LookupHost { addresses })
}

impl Iterator for LookupHost {
    type Item = SocketAddr;
    fn next(&mut self) -> Option<SocketAddr> {
        self.addresses.pop_front().map(|addr| from_netc(&addr))
    }
}

impl TryFrom<&str> for LookupHost {
    type Error = io::Error;

    fn try_from(host_port: &str) -> io::Result<LookupHost> {
        let (host, port_str) = host_port
            .rsplit_once(':')
            .ok_or(moto_rt::E_INVALID_ARGUMENT)
            .map_err(map_motor_error)?;
        let port: u16 =
            port_str.parse().map_err(|_| moto_rt::E_INVALID_ARGUMENT).map_err(map_motor_error)?;
        (host, port).try_into()
    }
}

impl<'a> TryFrom<(&'a str, u16)> for LookupHost {
    type Error = io::Error;

    fn try_from(host_port: (&'a str, u16)) -> io::Result<LookupHost> {
        let (host, port) = host_port;

        let (_port, addresses) = moto_rt::net::lookup_host(host, port).map_err(map_motor_error)?;
        Ok(LookupHost { addresses })
    }
}

fn into_netc(addr: &SocketAddr) -> netc::sockaddr {
    match addr {
        V4(addr4) => netc::sockaddr { v4: (*addr4).into() },
        V6(addr6) => netc::sockaddr { v6: (*addr6).into() },
    }
}

fn from_netc(addr: &netc::sockaddr) -> SocketAddr {
    // SAFETY: all variants of union netc::sockaddr have `sin_family` at the same offset.
    let family = unsafe { addr.v4.sin_family };
    match family {
        netc::AF_INET => SocketAddr::V4(crate::net::SocketAddrV4::from(unsafe { addr.v4 })),
        netc::AF_INET6 => SocketAddr::V6(crate::net::SocketAddrV6::from(unsafe { addr.v6 })),
        _ => panic!("bad sin_family {family}"),
    }
}

impl AsInner<FileDesc> for Socket {
    #[inline]
    fn as_inner(&self) -> &FileDesc {
        &self.0
    }
}

impl IntoInner<FileDesc> for Socket {
    fn into_inner(self) -> FileDesc {
        self.0
    }
}

impl FromInner<FileDesc> for Socket {
    fn from_inner(file_desc: FileDesc) -> Self {
        Self(file_desc)
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
        Self(FromRawFd::from_raw_fd(raw_fd))
    }
}

impl AsInner<Socket> for TcpStream {
    #[inline]
    fn as_inner(&self) -> &Socket {
        &self.inner
    }
}

impl FromInner<Socket> for TcpStream {
    fn from_inner(socket: Socket) -> TcpStream {
        TcpStream { inner: socket }
    }
}

impl FromInner<Socket> for TcpListener {
    fn from_inner(socket: Socket) -> TcpListener {
        TcpListener { inner: socket }
    }
}

impl FromInner<Socket> for UdpSocket {
    fn from_inner(socket: Socket) -> UdpSocket {
        UdpSocket { inner: socket }
    }
}
