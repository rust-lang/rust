use crate::fmt;
use crate::io::{self, IoSlice, IoSliceMut};
use crate::net::{SocketAddr, Shutdown, Ipv4Addr, Ipv6Addr, ToSocketAddrs};
use crate::time::Duration;
use crate::sys::{unsupported, Void, sgx_ineffective, AsInner, FromInner, IntoInner, TryIntoInner};
use crate::sys::fd::FileDesc;
use crate::convert::TryFrom;
use crate::error;
use crate::sync::Arc;

use super::abi::usercalls;

const DEFAULT_FAKE_TTL: u32 = 64;

#[derive(Debug, Clone)]
pub struct Socket {
    inner: Arc<FileDesc>,
    local_addr: Option<String>,
}

impl Socket {
    fn new(fd: usercalls::raw::Fd, local_addr: String) -> Socket {
        Socket { inner: Arc::new(FileDesc::new(fd)), local_addr: Some(local_addr) }
    }
}

impl AsInner<FileDesc> for Socket {
    fn as_inner(&self) -> &FileDesc { &self.inner }
}

impl TryIntoInner<FileDesc> for Socket {
    fn try_into_inner(self) -> Result<FileDesc, Socket> {
        let Socket { inner, local_addr } = self;
        Arc::try_unwrap(inner).map_err(|inner| Socket { inner, local_addr } )
    }
}

impl FromInner<FileDesc> for Socket {
    fn from_inner(inner: FileDesc) -> Socket {
        Socket { inner: Arc::new(inner), local_addr: None }
    }
}

#[derive(Clone)]
pub struct TcpStream {
    inner: Socket,
    peer_addr: Option<String>,
}

impl fmt::Debug for TcpStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut res = f.debug_struct("TcpStream");

        if let Some(ref addr) = self.inner.local_addr {
            res.field("addr", addr);
        }

        if let Some(ref peer) = self.peer_addr {
            res.field("peer", peer);
        }

        res.field("fd", &self.inner.inner.as_inner())
            .finish()
    }
}

fn io_err_to_addr(result: io::Result<&SocketAddr>) -> io::Result<String> {
    match result {
        Ok(saddr) => Ok(saddr.to_string()),
        // need to downcast twice because io::Error::into_inner doesn't return the original
        // value if the conversion fails
        Err(e) => if e.get_ref().and_then(|e| e.downcast_ref::<NonIpSockAddr>()).is_some() {
            Ok(e.into_inner().unwrap().downcast::<NonIpSockAddr>().unwrap().host)
        } else {
            Err(e)
        }
    }
}

fn addr_to_sockaddr(addr: &Option<String>) -> io::Result<SocketAddr> {
    addr.as_ref()
        .ok_or(io::ErrorKind::AddrNotAvailable)?
        .to_socket_addrs()
        // unwrap OK: if an iterator is returned, we're guaranteed to get exactly one entry
        .map(|mut it| it.next().unwrap())
}

impl TcpStream {
    pub fn connect(addr: io::Result<&SocketAddr>) -> io::Result<TcpStream> {
        let addr = io_err_to_addr(addr)?;
        let (fd, local_addr, peer_addr) = usercalls::connect_stream(&addr)?;
        Ok(TcpStream { inner: Socket::new(fd, local_addr), peer_addr: Some(peer_addr) })
    }

    pub fn connect_timeout(addr: &SocketAddr, dur: Duration) -> io::Result<TcpStream> {
        if dur == Duration::default() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput,
                                      "cannot set a 0 duration timeout"));
        }
        Self::connect(Ok(addr)) // FIXME: ignoring timeout
    }

    pub fn set_read_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        match dur {
            Some(dur) if dur == Duration::default() => {
                return Err(io::Error::new(io::ErrorKind::InvalidInput,
                                          "cannot set a 0 duration timeout"));
            }
            _ => sgx_ineffective(())
        }
    }

    pub fn set_write_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        match dur {
            Some(dur) if dur == Duration::default() => {
                return Err(io::Error::new(io::ErrorKind::InvalidInput,
                                          "cannot set a 0 duration timeout"));
            }
            _ => sgx_ineffective(())
        }
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        sgx_ineffective(None)
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        sgx_ineffective(None)
    }

    pub fn peek(&self, _: &mut [u8]) -> io::Result<usize> {
        Ok(0)
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.inner.read(buf)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.inner.inner.read_vectored(bufs)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.inner.inner.write(buf)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.inner.inner.write_vectored(bufs)
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        addr_to_sockaddr(&self.peer_addr)
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        addr_to_sockaddr(&self.inner.local_addr)
    }

    pub fn shutdown(&self, _: Shutdown) -> io::Result<()> {
        sgx_ineffective(())
    }

    pub fn duplicate(&self) -> io::Result<TcpStream> {
        Ok(self.clone())
    }

    pub fn set_nodelay(&self, _: bool) -> io::Result<()> {
        sgx_ineffective(())
    }

    pub fn nodelay(&self) -> io::Result<bool> {
        sgx_ineffective(false)
    }

    pub fn set_ttl(&self, _: u32) -> io::Result<()> {
        sgx_ineffective(())
    }

    pub fn ttl(&self) -> io::Result<u32> {
        sgx_ineffective(DEFAULT_FAKE_TTL)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        Ok(None)
    }

    pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
        sgx_ineffective(())
    }
}

impl AsInner<Socket> for TcpStream {
    fn as_inner(&self) -> &Socket { &self.inner }
}

// `Inner` includes `peer_addr` so that a `TcpStream` maybe correctly
// reconstructed if `Socket::try_into_inner` fails.
impl IntoInner<(Socket, Option<String>)> for TcpStream {
    fn into_inner(self) -> (Socket, Option<String>) {
        (self.inner, self.peer_addr)
    }
}

impl FromInner<(Socket, Option<String>)> for TcpStream {
    fn from_inner((inner, peer_addr): (Socket, Option<String>)) -> TcpStream {
        TcpStream { inner, peer_addr }
    }
}

#[derive(Clone)]
pub struct TcpListener {
    inner: Socket,
}

impl fmt::Debug for TcpListener {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut res = f.debug_struct("TcpListener");

        if let Some(ref addr) = self.inner.local_addr {
            res.field("addr", addr);
        }

        res.field("fd", &self.inner.inner.as_inner())
            .finish()
    }
}

impl TcpListener {
    pub fn bind(addr: io::Result<&SocketAddr>) -> io::Result<TcpListener> {
        let addr = io_err_to_addr(addr)?;
        let (fd, local_addr) = usercalls::bind_stream(&addr)?;
        Ok(TcpListener { inner: Socket::new(fd, local_addr) })
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        addr_to_sockaddr(&self.inner.local_addr)
    }

    pub fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        let (fd, local_addr, peer_addr) = usercalls::accept_stream(self.inner.inner.raw())?;
        let peer_addr = Some(peer_addr);
        let ret_peer = addr_to_sockaddr(&peer_addr).unwrap_or_else(|_| ([0; 4], 0).into());
        Ok((TcpStream { inner: Socket::new(fd, local_addr), peer_addr }, ret_peer))
    }

    pub fn duplicate(&self) -> io::Result<TcpListener> {
        Ok(self.clone())
    }

    pub fn set_ttl(&self, _: u32) -> io::Result<()> {
        sgx_ineffective(())
    }

    pub fn ttl(&self) -> io::Result<u32> {
        sgx_ineffective(DEFAULT_FAKE_TTL)
    }

    pub fn set_only_v6(&self, _: bool) -> io::Result<()> {
        sgx_ineffective(())
    }

    pub fn only_v6(&self) -> io::Result<bool> {
        sgx_ineffective(false)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        Ok(None)
    }

    pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
        sgx_ineffective(())
    }
}

impl AsInner<Socket> for TcpListener {
    fn as_inner(&self) -> &Socket { &self.inner }
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

pub struct UdpSocket(Void);

impl UdpSocket {
    pub fn bind(_: io::Result<&SocketAddr>) -> io::Result<UdpSocket> {
        unsupported()
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        match self.0 {}
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        match self.0 {}
    }

    pub fn recv_from(&self, _: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        match self.0 {}
    }

    pub fn peek_from(&self, _: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        match self.0 {}
    }

    pub fn send_to(&self, _: &[u8], _: &SocketAddr) -> io::Result<usize> {
        match self.0 {}
    }

    pub fn duplicate(&self) -> io::Result<UdpSocket> {
        match self.0 {}
    }

    pub fn set_read_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        match self.0 {}
    }

    pub fn set_write_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        match self.0 {}
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        match self.0 {}
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        match self.0 {}
    }

    pub fn set_broadcast(&self, _: bool) -> io::Result<()> {
        match self.0 {}
    }

    pub fn broadcast(&self) -> io::Result<bool> {
        match self.0 {}
    }

    pub fn set_multicast_loop_v4(&self, _: bool) -> io::Result<()> {
        match self.0 {}
    }

    pub fn multicast_loop_v4(&self) -> io::Result<bool> {
        match self.0 {}
    }

    pub fn set_multicast_ttl_v4(&self, _: u32) -> io::Result<()> {
        match self.0 {}
    }

    pub fn multicast_ttl_v4(&self) -> io::Result<u32> {
        match self.0 {}
    }

    pub fn set_multicast_loop_v6(&self, _: bool) -> io::Result<()> {
        match self.0 {}
    }

    pub fn multicast_loop_v6(&self) -> io::Result<bool> {
        match self.0 {}
    }

    pub fn join_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr)
                         -> io::Result<()> {
        match self.0 {}
    }

    pub fn join_multicast_v6(&self, _: &Ipv6Addr, _: u32)
                         -> io::Result<()> {
        match self.0 {}
    }

    pub fn leave_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr)
                          -> io::Result<()> {
        match self.0 {}
    }

    pub fn leave_multicast_v6(&self, _: &Ipv6Addr, _: u32)
                          -> io::Result<()> {
        match self.0 {}
    }

    pub fn set_ttl(&self, _: u32) -> io::Result<()> {
        match self.0 {}
    }

    pub fn ttl(&self) -> io::Result<u32> {
        match self.0 {}
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        match self.0 {}
    }

    pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
        match self.0 {}
    }

    pub fn recv(&self, _: &mut [u8]) -> io::Result<usize> {
        match self.0 {}
    }

    pub fn peek(&self, _: &mut [u8]) -> io::Result<usize> {
        match self.0 {}
    }

    pub fn send(&self, _: &[u8]) -> io::Result<usize> {
        match self.0 {}
    }

    pub fn connect(&self, _: io::Result<&SocketAddr>) -> io::Result<()> {
        match self.0 {}
    }
}

impl fmt::Debug for UdpSocket {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {}
    }
}

#[derive(Debug)]
pub struct NonIpSockAddr {
    host: String
}

impl error::Error for NonIpSockAddr {
    fn description(&self) -> &str {
        "Failed to convert address to SocketAddr"
    }
}

impl fmt::Display for NonIpSockAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Failed to convert address to SocketAddr: {}", self.host)
    }
}

pub struct LookupHost(Void);

impl LookupHost {
    fn new(host: String) -> io::Result<LookupHost> {
        Err(io::Error::new(io::ErrorKind::Other, NonIpSockAddr { host }))
    }

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

    fn try_from(v: &str) -> io::Result<LookupHost> {
        LookupHost::new(v.to_owned())
    }
}

impl<'a> TryFrom<(&'a str, u16)> for LookupHost {
    type Error = io::Error;

    fn try_from((host, port): (&'a str, u16)) -> io::Result<LookupHost> {
        LookupHost::new(format!("{}:{}", host, port))
    }
}

#[allow(bad_style)]
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
    pub struct sockaddr {
    }

    pub type socklen_t = usize;
}
