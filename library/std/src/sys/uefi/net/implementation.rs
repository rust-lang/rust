use super::tcp as uefi_tcp;
use crate::fmt;
use crate::io::{self, IoSlice, IoSliceMut};
use crate::net::{Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr};
use crate::sys::unsupported;
use crate::time::Duration;

pub struct TcpStream {
    inner: uefi_tcp::TcpProtocol,
}

impl TcpStream {
    fn new(inner: uefi_tcp::TcpProtocol) -> Self {
        Self { inner }
    }

    pub fn connect(_: io::Result<&SocketAddr>) -> io::Result<TcpStream> {
        todo!()
    }

    pub fn connect_timeout(_: &SocketAddr, _: Duration) -> io::Result<TcpStream> {
        todo!()
    }

    pub fn set_read_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        unimplemented!()
    }

    pub fn set_write_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        unimplemented!()
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        unimplemented!()
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        unimplemented!()
    }

    pub fn peek(&self, _: &mut [u8]) -> io::Result<usize> {
        unimplemented!()
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.inner.read_vectored(bufs)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        true
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.inner.write_vectored(bufs)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        true
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        todo!()
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        todo!()
    }

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        self.inner.shutdown(how)
    }

    pub fn duplicate(&self) -> io::Result<TcpStream> {
        unimplemented!()
    }

    // Seems to be similar to abort_on_close option in `EFI_TCP6_PROTOCOL->Close()`
    pub fn set_linger(&self, _: Option<Duration>) -> io::Result<()> {
        todo!()
    }

    pub fn linger(&self) -> io::Result<Option<Duration>> {
        todo!()
    }

    // Seems to be similar to `EFI_TCP6_OPTION->EnableNagle`
    pub fn set_nodelay(&self, _: bool) -> io::Result<()> {
        todo!()
    }

    pub fn nodelay(&self) -> io::Result<bool> {
        todo!()
    }

    pub fn set_ttl(&self, _: u32) -> io::Result<()> {
        unimplemented!()
    }

    pub fn ttl(&self) -> io::Result<u32> {
        unimplemented!()
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        unimplemented!()
    }

    pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
        todo!()
    }
}

impl fmt::Debug for TcpStream {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

pub struct TcpListener {
    inner: uefi_tcp::TcpProtocol,
}

impl TcpListener {
    fn new(inner: uefi_tcp::TcpProtocol) -> Self {
        Self { inner }
    }

    pub fn bind(addr: io::Result<&SocketAddr>) -> io::Result<TcpListener> {
        let addr = addr?;
        Ok(Self::new(uefi_tcp::TcpProtocol::bind(addr)?))
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        todo!()
    }

    pub fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        let (stream, socket_addr) = self.inner.accept()?;
        Ok((TcpStream::new(stream), socket_addr))
    }

    pub fn duplicate(&self) -> io::Result<TcpListener> {
        unimplemented!()
    }

    pub fn set_ttl(&self, _: u32) -> io::Result<()> {
        unimplemented!()
    }

    pub fn ttl(&self) -> io::Result<u32> {
        unimplemented!()
    }

    pub fn set_only_v6(&self, _: bool) -> io::Result<()> {
        unimplemented!()
    }

    pub fn only_v6(&self) -> io::Result<bool> {
        Ok(false)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        unimplemented!()
    }

    // Internally TCP6 Protocol is nonblocking
    pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
        todo!()
    }
}

impl fmt::Debug for TcpListener {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

pub struct UdpSocket {}

impl UdpSocket {
    pub fn bind(_: io::Result<&SocketAddr>) -> io::Result<UdpSocket> {
        unimplemented!()
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        unimplemented!()
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        unimplemented!()
    }

    pub fn recv_from(&self, _: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        unimplemented!()
    }

    pub fn peek_from(&self, _: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        unimplemented!()
    }

    pub fn send_to(&self, _: &[u8], _: &SocketAddr) -> io::Result<usize> {
        unimplemented!()
    }

    pub fn duplicate(&self) -> io::Result<UdpSocket> {
        unimplemented!()
    }

    pub fn set_read_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        unimplemented!()
    }

    pub fn set_write_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        unimplemented!()
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        unimplemented!()
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        unimplemented!()
    }

    pub fn set_broadcast(&self, _: bool) -> io::Result<()> {
        unimplemented!()
    }

    pub fn broadcast(&self) -> io::Result<bool> {
        unimplemented!()
    }

    pub fn set_multicast_loop_v4(&self, _: bool) -> io::Result<()> {
        unimplemented!()
    }

    pub fn multicast_loop_v4(&self) -> io::Result<bool> {
        unimplemented!()
    }

    pub fn set_multicast_ttl_v4(&self, _: u32) -> io::Result<()> {
        unimplemented!()
    }

    pub fn multicast_ttl_v4(&self) -> io::Result<u32> {
        unimplemented!()
    }

    pub fn set_multicast_loop_v6(&self, _: bool) -> io::Result<()> {
        unimplemented!()
    }

    pub fn multicast_loop_v6(&self) -> io::Result<bool> {
        unimplemented!()
    }

    pub fn join_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr) -> io::Result<()> {
        unimplemented!()
    }

    pub fn join_multicast_v6(&self, _: &Ipv6Addr, _: u32) -> io::Result<()> {
        unimplemented!()
    }

    pub fn leave_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr) -> io::Result<()> {
        unimplemented!()
    }

    pub fn leave_multicast_v6(&self, _: &Ipv6Addr, _: u32) -> io::Result<()> {
        unimplemented!()
    }

    pub fn set_ttl(&self, _: u32) -> io::Result<()> {
        unimplemented!()
    }

    pub fn ttl(&self) -> io::Result<u32> {
        unimplemented!()
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        unimplemented!()
    }

    pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
        unimplemented!()
    }

    pub fn recv(&self, _: &mut [u8]) -> io::Result<usize> {
        unimplemented!()
    }

    pub fn peek(&self, _: &mut [u8]) -> io::Result<usize> {
        unimplemented!()
    }

    pub fn send(&self, _: &[u8]) -> io::Result<usize> {
        unimplemented!()
    }

    pub fn connect(&self, _: io::Result<&SocketAddr>) -> io::Result<()> {
        unimplemented!()
    }
}

impl fmt::Debug for UdpSocket {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unimplemented!()
    }
}

pub struct LookupHost {}

impl LookupHost {
    pub fn port(&self) -> u16 {
        unimplemented!()
    }
}

impl Iterator for LookupHost {
    type Item = SocketAddr;
    fn next(&mut self) -> Option<SocketAddr> {
        unimplemented!()
    }
}

impl TryFrom<&str> for LookupHost {
    type Error = io::Error;

    fn try_from(_v: &str) -> io::Result<LookupHost> {
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
        pub sin_family: sa_family_t,
        pub sin_port: u16,
        pub sin_addr: in_addr,
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
    pub struct in6_addr {
        pub s6_addr: [u8; 16],
    }

    #[derive(Copy, Clone)]
    pub struct sockaddr {}

    pub type socklen_t = usize;
}
