use crate::convert::TryFrom;
use crate::fmt;
use crate::io::{self, ErrorKind, IoSlice, IoSliceMut};
use crate::net::{Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr};
use crate::str;
use crate::sys::hermit::abi;
use crate::sys::{unsupported, Void};
use crate::time::Duration;

/// Checks whether the HermitCore's socket interface has been started already, and
/// if not, starts it.
pub fn init() -> io::Result<()> {
    if abi::network_init() < 0 {
        return Err(io::Error::new(ErrorKind::Other, "Unable to initialize network interface"));
    }

    Ok(())
}

pub struct TcpStream(abi::Handle);

impl TcpStream {
    pub fn connect(addr: io::Result<&SocketAddr>) -> io::Result<TcpStream> {
        let addr = addr?;

        match abi::tcpstream::connect(addr.ip().to_string().as_bytes(), addr.port(), None) {
            Ok(handle) => Ok(TcpStream(handle)),
            _ => {
                Err(io::Error::new(ErrorKind::Other, "Unable to initiate a connection on a socket"))
            }
        }
    }

    pub fn connect_timeout(saddr: &SocketAddr, duration: Duration) -> io::Result<TcpStream> {
        match abi::tcpstream::connect(
            saddr.ip().to_string().as_bytes(),
            saddr.port(),
            Some(duration.as_millis() as u64),
        ) {
            Ok(handle) => Ok(TcpStream(handle)),
            _ => {
                Err(io::Error::new(ErrorKind::Other, "Unable to initiate a connection on a socket"))
            }
        }
    }

    pub fn set_read_timeout(&self, duration: Option<Duration>) -> io::Result<()> {
        abi::tcpstream::set_read_timeout(self.0, duration.map(|d| d.as_millis() as u64))
            .map_err(|_| io::Error::new(ErrorKind::Other, "Unable to set timeout value"))
    }

    pub fn set_write_timeout(&self, duration: Option<Duration>) -> io::Result<()> {
        abi::tcpstream::set_write_timeout(self.0, duration.map(|d| d.as_millis() as u64))
            .map_err(|_| io::Error::new(ErrorKind::Other, "Unable to set timeout value"))
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        let duration = abi::tcpstream::get_read_timeout(self.0)
            .map_err(|_| io::Error::new(ErrorKind::Other, "Unable to determine timeout value"))?;

        Ok(duration.map(|d| Duration::from_millis(d)))
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        let duration = abi::tcpstream::get_write_timeout(self.0)
            .map_err(|_| io::Error::new(ErrorKind::Other, "Unable to determine timeout value"))?;

        Ok(duration.map(|d| Duration::from_millis(d)))
    }

    pub fn peek(&self, buf: &mut [u8]) -> io::Result<usize> {
        abi::tcpstream::peek(self.0, buf)
            .map_err(|_| io::Error::new(ErrorKind::Other, "set_nodelay failed"))
    }

    pub fn read(&self, buffer: &mut [u8]) -> io::Result<usize> {
        self.read_vectored(&mut [IoSliceMut::new(buffer)])
    }

    pub fn read_vectored(&self, ioslice: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let mut size: usize = 0;

        for i in ioslice.iter_mut() {
            let mut pos: usize = 0;

            while pos < i.len() {
                let ret = abi::tcpstream::read(self.0, &mut i[pos..])
                    .map_err(|_| io::Error::new(ErrorKind::Other, "Unable to read on socket"))?;

                if ret == 0 {
                    return Ok(size);
                } else {
                    size += ret;
                    pos += ret;
                }
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
            size += abi::tcpstream::write(self.0, i)
                .map_err(|_| io::Error::new(ErrorKind::Other, "Unable to write on socket"))?;
        }

        Ok(size)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        true
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        Err(io::Error::new(ErrorKind::Other, "peer_addr isn't supported"))
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        Err(io::Error::new(ErrorKind::Other, "socket_addr isn't supported"))
    }

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        abi::tcpstream::shutdown(self.0, how as i32)
            .map_err(|_| io::Error::new(ErrorKind::Other, "unable to shutdown socket"))
    }

    pub fn duplicate(&self) -> io::Result<TcpStream> {
        let handle = abi::tcpstream::duplicate(self.0)
            .map_err(|_| io::Error::new(ErrorKind::Other, "unable to duplicate stream"))?;

        Ok(TcpStream(handle))
    }

    pub fn set_nodelay(&self, mode: bool) -> io::Result<()> {
        abi::tcpstream::set_nodelay(self.0, mode)
            .map_err(|_| io::Error::new(ErrorKind::Other, "set_nodelay failed"))
    }

    pub fn nodelay(&self) -> io::Result<bool> {
        abi::tcpstream::nodelay(self.0)
            .map_err(|_| io::Error::new(ErrorKind::Other, "nodelay failed"))
    }

    pub fn set_ttl(&self, tll: u32) -> io::Result<()> {
        abi::tcpstream::set_tll(self.0, tll)
            .map_err(|_| io::Error::new(ErrorKind::Other, "unable to set TTL"))
    }

    pub fn ttl(&self) -> io::Result<u32> {
        abi::tcpstream::get_tll(self.0)
            .map_err(|_| io::Error::new(ErrorKind::Other, "unable to get TTL"))
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        Err(io::Error::new(ErrorKind::Other, "take_error isn't supported"))
    }

    pub fn set_nonblocking(&self, mode: bool) -> io::Result<()> {
        abi::tcpstream::set_nonblocking(self.0, mode)
            .map_err(|_| io::Error::new(ErrorKind::Other, "unable to set blocking mode"))
    }
}

impl Drop for TcpStream {
    fn drop(&mut self) {
        let _ = abi::tcpstream::close(self.0);
    }
}

impl fmt::Debug for TcpStream {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

pub struct TcpListener(abi::Handle);

impl TcpListener {
    pub fn bind(_: io::Result<&SocketAddr>) -> io::Result<TcpListener> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
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

pub struct UdpSocket(abi::Handle);

impl UdpSocket {
    pub fn bind(_: io::Result<&SocketAddr>) -> io::Result<UdpSocket> {
        Err(io::Error::new(ErrorKind::Other, "not supported"))
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
