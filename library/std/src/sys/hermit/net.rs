use crate::convert::TryFrom;
use crate::fmt;
use crate::io::{self, ErrorKind, IoSlice, IoSliceMut};
use crate::net::{IpAddr, Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr};
use crate::str;
use crate::sync::Arc;
use crate::sys::hermit::abi;
use crate::sys::hermit::abi::IpAddress::{Ipv4, Ipv6};
use crate::sys::unsupported;
use crate::sys_common::AsInner;
use crate::time::Duration;

/// Checks whether the HermitCore's socket interface has been started already, and
/// if not, starts it.
pub fn init() -> io::Result<()> {
    if abi::network_init() < 0 {
        return Err(io::Error::new_const(
            ErrorKind::Uncategorized,
            &"Unable to initialize network interface",
        ));
    }

    Ok(())
}

#[derive(Debug, Clone)]
pub struct Socket(abi::Handle);

impl AsInner<abi::Handle> for Socket {
    fn as_inner(&self) -> &abi::Handle {
        &self.0
    }
}

impl Drop for Socket {
    fn drop(&mut self) {
        let _ = abi::tcpstream::close(self.0);
    }
}

// Arc is used to count the number of used sockets.
// Only if all sockets are released, the drop
// method will close the socket.
#[derive(Clone)]
pub struct TcpStream(Arc<Socket>);

impl TcpStream {
    pub fn connect(addr: io::Result<&SocketAddr>) -> io::Result<TcpStream> {
        let addr = addr?;

        match abi::tcpstream::connect(addr.ip().to_string().as_bytes(), addr.port(), None) {
            Ok(handle) => Ok(TcpStream(Arc::new(Socket(handle)))),
            _ => Err(io::Error::new_const(
                ErrorKind::Uncategorized,
                &"Unable to initiate a connection on a socket",
            )),
        }
    }

    pub fn connect_timeout(saddr: &SocketAddr, duration: Duration) -> io::Result<TcpStream> {
        match abi::tcpstream::connect(
            saddr.ip().to_string().as_bytes(),
            saddr.port(),
            Some(duration.as_millis() as u64),
        ) {
            Ok(handle) => Ok(TcpStream(Arc::new(Socket(handle)))),
            _ => Err(io::Error::new_const(
                ErrorKind::Uncategorized,
                &"Unable to initiate a connection on a socket",
            )),
        }
    }

    pub fn set_read_timeout(&self, duration: Option<Duration>) -> io::Result<()> {
        abi::tcpstream::set_read_timeout(*self.0.as_inner(), duration.map(|d| d.as_millis() as u64))
            .map_err(|_| {
                io::Error::new_const(ErrorKind::Uncategorized, &"Unable to set timeout value")
            })
    }

    pub fn set_write_timeout(&self, duration: Option<Duration>) -> io::Result<()> {
        abi::tcpstream::set_write_timeout(
            *self.0.as_inner(),
            duration.map(|d| d.as_millis() as u64),
        )
        .map_err(|_| io::Error::new_const(ErrorKind::Uncategorized, &"Unable to set timeout value"))
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        let duration = abi::tcpstream::get_read_timeout(*self.0.as_inner()).map_err(|_| {
            io::Error::new_const(ErrorKind::Uncategorized, &"Unable to determine timeout value")
        })?;

        Ok(duration.map(|d| Duration::from_millis(d)))
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        let duration = abi::tcpstream::get_write_timeout(*self.0.as_inner()).map_err(|_| {
            io::Error::new_const(ErrorKind::Uncategorized, &"Unable to determine timeout value")
        })?;

        Ok(duration.map(|d| Duration::from_millis(d)))
    }

    pub fn peek(&self, buf: &mut [u8]) -> io::Result<usize> {
        abi::tcpstream::peek(*self.0.as_inner(), buf)
            .map_err(|_| io::Error::new_const(ErrorKind::Uncategorized, &"peek failed"))
    }

    pub fn read(&self, buffer: &mut [u8]) -> io::Result<usize> {
        self.read_vectored(&mut [IoSliceMut::new(buffer)])
    }

    pub fn read_vectored(&self, ioslice: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let mut size: usize = 0;

        for i in ioslice.iter_mut() {
            let ret = abi::tcpstream::read(*self.0.as_inner(), &mut i[0..]).map_err(|_| {
                io::Error::new_const(ErrorKind::Uncategorized, &"Unable to read on socket")
            })?;

            if ret != 0 {
                size += ret;
            }
        }

        Ok(size)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        true
    }

    pub fn write(&self, buffer: &[u8]) -> io::Result<usize> {
        self.write_vectored(&[IoSlice::new(buffer)])
    }

    pub fn write_vectored(&self, ioslice: &[IoSlice<'_>]) -> io::Result<usize> {
        let mut size: usize = 0;

        for i in ioslice.iter() {
            size += abi::tcpstream::write(*self.0.as_inner(), i).map_err(|_| {
                io::Error::new_const(ErrorKind::Uncategorized, &"Unable to write on socket")
            })?;
        }

        Ok(size)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        true
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        let (ipaddr, port) = abi::tcpstream::peer_addr(*self.0.as_inner())
            .map_err(|_| io::Error::new_const(ErrorKind::Uncategorized, &"peer_addr failed"))?;

        let saddr = match ipaddr {
            Ipv4(ref addr) => SocketAddr::new(IpAddr::V4(Ipv4Addr::from(addr.0)), port),
            Ipv6(ref addr) => SocketAddr::new(IpAddr::V6(Ipv6Addr::from(addr.0)), port),
            _ => {
                return Err(io::Error::new_const(ErrorKind::Uncategorized, &"peer_addr failed"));
            }
        };

        Ok(saddr)
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        unsupported()
    }

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        abi::tcpstream::shutdown(*self.0.as_inner(), how as i32).map_err(|_| {
            io::Error::new_const(ErrorKind::Uncategorized, &"unable to shutdown socket")
        })
    }

    pub fn duplicate(&self) -> io::Result<TcpStream> {
        Ok(self.clone())
    }

    pub fn set_linger(&self, _linger: Option<Duration>) -> io::Result<()> {
        unsupported()
    }

    pub fn linger(&self) -> io::Result<Option<Duration>> {
        unsupported()
    }

    pub fn set_nodelay(&self, mode: bool) -> io::Result<()> {
        abi::tcpstream::set_nodelay(*self.0.as_inner(), mode)
            .map_err(|_| io::Error::new_const(ErrorKind::Uncategorized, &"set_nodelay failed"))
    }

    pub fn nodelay(&self) -> io::Result<bool> {
        abi::tcpstream::nodelay(*self.0.as_inner())
            .map_err(|_| io::Error::new_const(ErrorKind::Uncategorized, &"nodelay failed"))
    }

    pub fn set_ttl(&self, tll: u32) -> io::Result<()> {
        abi::tcpstream::set_tll(*self.0.as_inner(), tll)
            .map_err(|_| io::Error::new_const(ErrorKind::Uncategorized, &"unable to set TTL"))
    }

    pub fn ttl(&self) -> io::Result<u32> {
        abi::tcpstream::get_tll(*self.0.as_inner())
            .map_err(|_| io::Error::new_const(ErrorKind::Uncategorized, &"unable to get TTL"))
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        unsupported()
    }

    pub fn set_nonblocking(&self, mode: bool) -> io::Result<()> {
        abi::tcpstream::set_nonblocking(*self.0.as_inner(), mode).map_err(|_| {
            io::Error::new_const(ErrorKind::Uncategorized, &"unable to set blocking mode")
        })
    }
}

impl fmt::Debug for TcpStream {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

#[derive(Clone)]
pub struct TcpListener(SocketAddr);

impl TcpListener {
    pub fn bind(addr: io::Result<&SocketAddr>) -> io::Result<TcpListener> {
        let addr = addr?;

        Ok(TcpListener(*addr))
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        Ok(self.0)
    }

    pub fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        let (handle, ipaddr, port) = abi::tcplistener::accept(self.0.port())
            .map_err(|_| io::Error::new_const(ErrorKind::Uncategorized, &"accept failed"))?;
        let saddr = match ipaddr {
            Ipv4(ref addr) => SocketAddr::new(IpAddr::V4(Ipv4Addr::from(addr.0)), port),
            Ipv6(ref addr) => SocketAddr::new(IpAddr::V6(Ipv6Addr::from(addr.0)), port),
            _ => {
                return Err(io::Error::new_const(ErrorKind::Uncategorized, &"accept failed"));
            }
        };

        Ok((TcpStream(Arc::new(Socket(handle))), saddr))
    }

    pub fn duplicate(&self) -> io::Result<TcpListener> {
        Ok(self.clone())
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

    pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
        unsupported()
    }
}

impl fmt::Debug for TcpListener {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

pub struct UdpSocket(abi::Handle);

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
}

impl fmt::Debug for UdpSocket {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
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
