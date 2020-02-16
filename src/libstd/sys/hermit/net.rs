use crate::convert::TryFrom;
use crate::fmt;
use crate::io::{self, IoSlice, IoSliceMut, ErrorKind};
use crate::net::{Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr};
use crate::str;
use crate::sync::atomic::{AtomicBool, Ordering};
use crate::sys::hermit::abi;
use crate::sys::{unsupported, Void};
use crate::time::Duration;

/// Checks whether the HermitCore's socket interface has been started already, and
/// if not, starts it.
fn init() -> io::Result<()> {
    static START: AtomicBool = AtomicBool::new(false);

    if START.swap(true, Ordering::SeqCst) == false {
        if abi::network_init() < 0 {
            return Err(io::Error::new(ErrorKind::Other, "Unable to initialize network interface"));
        }
    }

    Ok(())
}

pub struct TcpStream(i32);

impl TcpStream {
    pub fn connect(result_addr: io::Result<&SocketAddr>) -> io::Result<TcpStream> {
        init()?;

        // unpack socker address
        if result_addr.is_err() {
            return Err(result_addr.unwrap_err());
        }
        let _saddr = result_addr.unwrap();

        Ok(TcpStream(0))
    }

    pub fn connect_timeout(_: &SocketAddr, _: Duration) -> io::Result<TcpStream> {
        init()?;

		Ok(TcpStream(0))
    }

    pub fn set_read_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn set_write_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn peek(&self, _: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn read(&self, _: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn read_vectored(&self, _: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn write(&self, _: &[u8]) -> io::Result<usize> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn write_vectored(&self, _: &[IoSlice<'_>]) -> io::Result<usize> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn shutdown(&self, _: Shutdown) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn duplicate(&self) -> io::Result<TcpStream> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn set_nodelay(&self, _: bool) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn nodelay(&self) -> io::Result<bool> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn set_ttl(&self, _: u32) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn ttl(&self) -> io::Result<u32> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }
}

impl fmt::Debug for TcpStream {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

pub struct TcpListener(i32);

impl TcpListener {
    pub fn bind(_: io::Result<&SocketAddr>) -> io::Result<TcpListener> {
        init()?;

        Ok(TcpListener(0))
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn duplicate(&self) -> io::Result<TcpListener> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn set_ttl(&self, _: u32) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn ttl(&self) -> io::Result<u32> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn set_only_v6(&self, _: bool) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn only_v6(&self) -> io::Result<bool> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }
}

impl fmt::Debug for TcpListener {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

pub struct UdpSocket(i32);

impl UdpSocket {
    pub fn bind(_: io::Result<&SocketAddr>) -> io::Result<UdpSocket> {
        init()?;

        Ok(UdpSocket(0))
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn recv_from(&self, _: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn peek_from(&self, _: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn send_to(&self, _: &[u8], _: &SocketAddr) -> io::Result<usize> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn duplicate(&self) -> io::Result<UdpSocket> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn set_read_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn set_write_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn set_broadcast(&self, _: bool) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn broadcast(&self) -> io::Result<bool> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn set_multicast_loop_v4(&self, _: bool) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn multicast_loop_v4(&self) -> io::Result<bool> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn set_multicast_ttl_v4(&self, _: u32) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn multicast_ttl_v4(&self) -> io::Result<u32> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn set_multicast_loop_v6(&self, _: bool) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn multicast_loop_v6(&self) -> io::Result<bool> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn join_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn join_multicast_v6(&self, _: &Ipv6Addr, _: u32) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn leave_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn leave_multicast_v6(&self, _: &Ipv6Addr, _: u32) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn set_ttl(&self, _: u32) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn ttl(&self) -> io::Result<u32> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn recv(&self, _: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn peek(&self, _: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn send(&self, _: &[u8]) -> io::Result<usize> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }

    pub fn connect(&self, _: io::Result<&SocketAddr>) -> io::Result<()> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
    }
}

impl fmt::Debug for UdpSocket {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

pub struct LookupHost(Void);

impl LookupHost {
    pub fn port(&self) -> u16 {
        match self.0 {}
    }
}

impl Iterator for LookupHost {
    type Item = SocketAddr;
    fn next(&mut self) -> Option<SocketAddr> {
        match self.0 {}
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
        init()?;

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

    pub type socklen_t = usize;
}
