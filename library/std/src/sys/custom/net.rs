#![unstable(issue = "none", feature = "std_internals")]
#![allow(missing_docs)]

use crate::custom_os_impl;
use crate::fmt;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut};
use crate::net::{Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr};
use crate::time::Duration;

/// Inner content of [`crate::net::TcpStream`]
pub struct TcpStream(pub Box<dyn TcpStreamApi>);

/// Object-oriented manipulation of a [`TcpStream`]
pub trait TcpStreamApi: fmt::Debug {
    fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()>;
    fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()>;
    fn read_timeout(&self) -> io::Result<Option<Duration>>;
    fn write_timeout(&self) -> io::Result<Option<Duration>>;
    fn peek(&self, bytes: &mut [u8]) -> io::Result<usize>;
    fn read(&self, bytes: &mut [u8]) -> io::Result<usize>;
    fn read_buf(&self, _buf: BorrowedCursor<'_>) -> io::Result<()>;
    fn read_vectored(&self, slices: &mut [IoSliceMut<'_>]) -> io::Result<usize>;
    fn is_read_vectored(&self) -> bool;
    fn write(&self, bytes: &[u8]) -> io::Result<usize>;
    fn write_vectored(&self, slices: &[IoSlice<'_>]) -> io::Result<usize>;
    fn is_write_vectored(&self) -> bool;
    fn peer_addr(&self) -> io::Result<SocketAddr>;
    fn socket_addr(&self) -> io::Result<SocketAddr>;
    fn shutdown(&self, shutdown: Shutdown) -> io::Result<()>;
    fn duplicate(&self) -> io::Result<TcpStream>;
    fn set_linger(&self, linger: Option<Duration>) -> io::Result<()>;
    fn linger(&self) -> io::Result<Option<Duration>>;
    fn set_nodelay(&self, nodelay: bool) -> io::Result<()>;
    fn nodelay(&self) -> io::Result<bool>;
    fn set_ttl(&self, ttl: u32) -> io::Result<()>;
    fn ttl(&self) -> io::Result<u32>;
    fn take_error(&self) -> io::Result<Option<io::Error>>;
    fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()>;
}

impl TcpStream {
    pub fn connect(addr: io::Result<&SocketAddr>) -> io::Result<TcpStream> {
        custom_os_impl!(net, tcp_connect, addr?, None)
    }

    pub fn connect_timeout(addr: &SocketAddr, timeout: Duration) -> io::Result<TcpStream> {
        custom_os_impl!(net, tcp_connect, addr, Some(timeout))
    }
}

impl core::ops::Deref for TcpStream {
    type Target = dyn TcpStreamApi;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

/// Inner content of [`crate::net::TcpListener`]
pub struct TcpListener(pub Box<dyn TcpListenerApi>);

/// Object-oriented manipulation of a [`TcpListener`]
pub trait TcpListenerApi: fmt::Debug {
    fn socket_addr(&self) -> io::Result<SocketAddr>;
    fn accept(&self) -> io::Result<(TcpStream, SocketAddr)>;
    fn duplicate(&self) -> io::Result<TcpListener>;
    fn set_ttl(&self, ttl: u32) -> io::Result<()>;
    fn ttl(&self) -> io::Result<u32>;
    fn set_only_v6(&self, only_v6: bool) -> io::Result<()>;
    fn only_v6(&self) -> io::Result<bool>;
    fn take_error(&self) -> io::Result<Option<io::Error>>;
    fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()>;
}

impl TcpListener {
    pub fn bind(addr: io::Result<&SocketAddr>) -> io::Result<TcpListener> {
        custom_os_impl!(net, tcp_bind, addr?)
    }
}

impl core::ops::Deref for TcpListener {
    type Target = dyn TcpListenerApi;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

/// Inner content of [`crate::net::UdpSocket`]
pub struct UdpSocket(pub Box<dyn UdpSocketApi>);

/// Object-oriented manipulation of a [`UdpSocket`]
pub trait UdpSocketApi: fmt::Debug {
    fn peer_addr(&self) -> io::Result<SocketAddr>;
    fn socket_addr(&self) -> io::Result<SocketAddr>;
    fn recv_from(&self, _: &mut [u8]) -> io::Result<(usize, SocketAddr)>;
    fn peek_from(&self, _: &mut [u8]) -> io::Result<(usize, SocketAddr)>;
    fn send_to(&self, _: &[u8], _: &SocketAddr) -> io::Result<usize>;
    fn duplicate(&self) -> io::Result<UdpSocket>;
    fn set_read_timeout(&self, _: Option<Duration>) -> io::Result<()>;
    fn set_write_timeout(&self, _: Option<Duration>) -> io::Result<()>;
    fn read_timeout(&self) -> io::Result<Option<Duration>>;
    fn write_timeout(&self) -> io::Result<Option<Duration>>;
    fn set_broadcast(&self, _: bool) -> io::Result<()>;
    fn broadcast(&self) -> io::Result<bool>;
    fn set_multicast_loop_v4(&self, _: bool) -> io::Result<()>;
    fn multicast_loop_v4(&self) -> io::Result<bool>;
    fn set_multicast_ttl_v4(&self, _: u32) -> io::Result<()>;
    fn multicast_ttl_v4(&self) -> io::Result<u32>;
    fn set_multicast_loop_v6(&self, _: bool) -> io::Result<()>;
    fn multicast_loop_v6(&self) -> io::Result<bool>;
    fn join_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr) -> io::Result<()>;
    fn join_multicast_v6(&self, _: &Ipv6Addr, _: u32) -> io::Result<()>;
    fn leave_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr) -> io::Result<()>;
    fn leave_multicast_v6(&self, _: &Ipv6Addr, _: u32) -> io::Result<()>;
    fn set_ttl(&self, _: u32) -> io::Result<()>;
    fn ttl(&self) -> io::Result<u32>;
    fn take_error(&self) -> io::Result<Option<io::Error>>;
    fn set_nonblocking(&self, _: bool) -> io::Result<()>;
    fn recv(&self, _: &mut [u8]) -> io::Result<usize>;
    fn peek(&self, _: &mut [u8]) -> io::Result<usize>;
    fn send(&self, _: &[u8]) -> io::Result<usize>;
    fn connect(&self, _: io::Result<&SocketAddr>) -> io::Result<()>;
}

impl UdpSocket {
    pub fn bind(addr: io::Result<&SocketAddr>) -> io::Result<UdpSocket> {
        custom_os_impl!(net, udp_bind, addr?)
    }
}

impl core::ops::Deref for UdpSocket {
    type Target = dyn UdpSocketApi;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

/// Result of hostname/port parsing & resolution
pub struct LookupHost {
    addresses: Vec<SocketAddr>,
    port: u16,
    i: usize,
}

impl LookupHost {
    pub fn new(addresses: Vec<SocketAddr>, port: u16) -> Self {
        Self { addresses, port, i: 0 }
    }

    pub(crate) fn port(&self) -> u16 {
        self.port
    }
}

impl Iterator for LookupHost {
    type Item = SocketAddr;
    fn next(&mut self) -> Option<SocketAddr> {
        let retval = self.addresses.get(self.i)?;
        self.i += 1;
        Some(*retval)
    }
}

impl TryFrom<&str> for LookupHost {
    type Error = io::Error;

    fn try_from(v: &str) -> io::Result<LookupHost> {
        custom_os_impl!(net, lookup_str, v)
    }
}

impl<'a> TryFrom<(&'a str, u16)> for LookupHost {
    type Error = io::Error;

    fn try_from(v: (&'a str, u16)) -> io::Result<LookupHost> {
        custom_os_impl!(net, lookup_tuple, v)
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
        pub sin6_family: sa_family_t,
        pub sin6_port: u16,
        pub sin6_addr: in6_addr,
        pub sin6_flowinfo: u32,
        pub sin6_scope_id: u32,
    }

    #[derive(Copy, Clone)]
    pub struct sockaddr {}
}
