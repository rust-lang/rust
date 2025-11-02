use crate::error;
use crate::fmt::{self, Write};
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut};
use crate::net::{Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, ToSocketAddrs};
use crate::sync::Arc;
use crate::sys::abi::usercalls;
use crate::sys::fd::FileDesc;
use crate::sys::{AsInner, FromInner, IntoInner, TryIntoInner, sgx_ineffective, unsupported};
use crate::time::Duration;

const DEFAULT_FAKE_TTL: u32 = 64;
const DEFAULT_FAKE_HLIM: u8 = 64;

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
    #[inline]
    fn as_inner(&self) -> &FileDesc {
        &self.inner
    }
}

impl TryIntoInner<FileDesc> for Socket {
    fn try_into_inner(self) -> Result<FileDesc, Socket> {
        let Socket { inner, local_addr } = self;
        Arc::try_unwrap(inner).map_err(|inner| Socket { inner, local_addr })
    }
}

impl FromInner<(FileDesc, Option<String>)> for Socket {
    fn from_inner((inner, local_addr): (FileDesc, Option<String>)) -> Socket {
        Socket { inner: Arc::new(inner), local_addr }
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

        res.field("fd", &self.inner.inner.as_inner()).finish()
    }
}

/// Converts each address in `addr` into a hostname.
///
/// SGX doesn't support DNS resolution but rather accepts hostnames in
/// the same place as socket addresses. So, to make e.g.
/// ```rust
/// TcpStream::connect("example.com:80")`
/// ```
/// work, the DNS lookup returns a special error (`NonIpSockAddr`) instead,
/// which contains the hostname being looked up. When `.to_socket_addrs()`
/// fails, we inspect the error and try recover the hostname from it. If that
/// succeeds, we thus continue with the hostname.
///
/// This is a terrible hack and leads to buggy code. For instance, when users
/// use the result of `.to_socket_addrs()` in their own `ToSocketAddrs`
/// implementation to select from a list of possible URLs, the only URL used
/// will be that of the last item tried.
// FIXME: This is a terrible, terrible hack. Fixing this requires Fortanix to
// add a method for resolving addresses.
fn each_addr<A: ToSocketAddrs, F, T>(addr: A, mut f: F) -> io::Result<T>
where
    F: FnMut(&str) -> io::Result<T>,
{
    match addr.to_socket_addrs() {
        Ok(addrs) => {
            let mut last_err = None;
            let mut encoded = String::new();
            for addr in addrs {
                // Format the IP address as a string, reusing the buffer.
                encoded.clear();
                write!(encoded, "{}", &addr).unwrap();

                match f(&encoded) {
                    Ok(val) => return Ok(val),
                    Err(err) => last_err = Some(err),
                }
            }

            match last_err {
                Some(err) => Err(err),
                None => Err(io::Error::NO_ADDRESSES),
            }
        }
        Err(err) => match err.get_ref().and_then(|e| e.downcast_ref::<NonIpSockAddr>()) {
            Some(NonIpSockAddr { host }) => f(host),
            None => Err(err),
        },
    }
}

fn addr_to_sockaddr(addr: Option<&str>) -> io::Result<SocketAddr> {
    addr.ok_or(io::ErrorKind::AddrNotAvailable)?
        .to_socket_addrs()
        // unwrap OK: if an iterator is returned, we're guaranteed to get exactly one entry
        .map(|mut it| it.next().unwrap())
}

impl TcpStream {
    pub fn connect<A: ToSocketAddrs>(addr: A) -> io::Result<TcpStream> {
        each_addr(addr, |addr| {
            let (fd, local_addr, peer_addr) = usercalls::connect_stream(addr)?;
            Ok(TcpStream { inner: Socket::new(fd, local_addr), peer_addr: Some(peer_addr) })
        })
    }

    pub fn connect_timeout(addr: &SocketAddr, dur: Duration) -> io::Result<TcpStream> {
        if dur == Duration::default() {
            return Err(io::Error::ZERO_TIMEOUT);
        }
        Self::connect(addr) // FIXME: ignoring timeout
    }

    pub fn set_read_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        match dur {
            Some(dur) if dur == Duration::default() => {
                return Err(io::Error::ZERO_TIMEOUT);
            }
            _ => sgx_ineffective(()),
        }
    }

    pub fn set_write_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        match dur {
            Some(dur) if dur == Duration::default() => {
                return Err(io::Error::ZERO_TIMEOUT);
            }
            _ => sgx_ineffective(()),
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

    pub fn read_buf(&self, buf: BorrowedCursor<'_>) -> io::Result<()> {
        self.inner.inner.read_buf(buf)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.inner.inner.read_vectored(bufs)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        self.inner.inner.is_read_vectored()
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.inner.inner.write(buf)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.inner.inner.write_vectored(bufs)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        self.inner.inner.is_write_vectored()
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        addr_to_sockaddr(self.peer_addr.as_deref())
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        addr_to_sockaddr(self.inner.local_addr.as_deref())
    }

    pub fn shutdown(&self, _: Shutdown) -> io::Result<()> {
        sgx_ineffective(())
    }

    pub fn duplicate(&self) -> io::Result<TcpStream> {
        Ok(self.clone())
    }

    pub fn set_linger(&self, _: Option<Duration>) -> io::Result<()> {
        sgx_ineffective(())
    }

    pub fn linger(&self) -> io::Result<Option<Duration>> {
        sgx_ineffective(None)
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

    pub fn set_hop_limit_v6(&self, _: u8) -> io::Result<()> {
        sgx_ineffective(())
    }

    pub fn hop_limit_v6(&self) -> io::Result<u8> {
        sgx_ineffective(DEFAULT_FAKE_HLIM)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        Ok(None)
    }

    pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
        sgx_ineffective(())
    }
}

impl AsInner<Socket> for TcpStream {
    #[inline]
    fn as_inner(&self) -> &Socket {
        &self.inner
    }
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

        res.field("fd", &self.inner.inner.as_inner()).finish()
    }
}

impl TcpListener {
    pub fn bind<A: ToSocketAddrs>(addr: A) -> io::Result<TcpListener> {
        each_addr(addr, |addr| {
            let (fd, local_addr) = usercalls::bind_stream(addr)?;
            Ok(TcpListener { inner: Socket::new(fd, local_addr) })
        })
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        addr_to_sockaddr(self.inner.local_addr.as_deref())
    }

    pub fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        let (fd, local_addr, peer_addr) = usercalls::accept_stream(self.inner.inner.raw())?;
        let peer_addr = Some(peer_addr);
        let ret_peer =
            addr_to_sockaddr(peer_addr.as_deref()).unwrap_or_else(|_| ([0; 4], 0).into());
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

    pub fn set_hop_limit_v6(&self, _: u8) -> io::Result<()> {
        sgx_ineffective(())
    }

    pub fn hop_limit_v6(&self) -> io::Result<u8> {
        sgx_ineffective(DEFAULT_FAKE_HLIM)
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

pub struct UdpSocket(!);

impl UdpSocket {
    pub fn bind<A: ToSocketAddrs>(_: A) -> io::Result<UdpSocket> {
        unsupported()
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        self.0
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        self.0
    }

    pub fn recv_from(&self, _: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.0
    }

    pub fn peek_from(&self, _: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.0
    }

    pub fn send_to(&self, _: &[u8], _: &SocketAddr) -> io::Result<usize> {
        self.0
    }

    pub fn duplicate(&self) -> io::Result<UdpSocket> {
        self.0
    }

    pub fn set_read_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        self.0
    }

    pub fn set_write_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        self.0
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        self.0
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        self.0
    }

    pub fn set_broadcast(&self, _: bool) -> io::Result<()> {
        self.0
    }

    pub fn broadcast(&self) -> io::Result<bool> {
        self.0
    }

    pub fn set_multicast_loop_v4(&self, _: bool) -> io::Result<()> {
        self.0
    }

    pub fn multicast_loop_v4(&self) -> io::Result<bool> {
        self.0
    }

    pub fn set_multicast_ttl_v4(&self, _: u32) -> io::Result<()> {
        self.0
    }

    pub fn multicast_ttl_v4(&self) -> io::Result<u32> {
        self.0
    }

    pub fn set_multicast_loop_v6(&self, _: bool) -> io::Result<()> {
        self.0
    }

    pub fn multicast_loop_v6(&self) -> io::Result<bool> {
        self.0
    }

    pub fn join_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr) -> io::Result<()> {
        self.0
    }

    pub fn join_multicast_v6(&self, _: &Ipv6Addr, _: u32) -> io::Result<()> {
        self.0
    }

    pub fn leave_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr) -> io::Result<()> {
        self.0
    }

    pub fn leave_multicast_v6(&self, _: &Ipv6Addr, _: u32) -> io::Result<()> {
        self.0
    }

    pub fn set_ttl(&self, _: u32) -> io::Result<()> {
        self.0
    }

    pub fn ttl(&self) -> io::Result<u32> {
        self.0
    }

    pub fn set_hop_limit_v6(&self, _: u8) -> io::Result<()> {
        self.0
    }

    pub fn hop_limit_v6(&self) -> io::Result<u8> {
        self.0
    }

    pub fn set_multicast_hop_limit_v6(&self, _: u8) -> io::Result<()> {
        self.0
    }

    pub fn multicast_hop_limit_v6(&self) -> io::Result<u8> {
        self.0
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        self.0
    }

    pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
        self.0
    }

    pub fn recv(&self, _: &mut [u8]) -> io::Result<usize> {
        self.0
    }

    pub fn peek(&self, _: &mut [u8]) -> io::Result<usize> {
        self.0
    }

    pub fn send(&self, _: &[u8]) -> io::Result<usize> {
        self.0
    }

    pub fn connect<A: ToSocketAddrs>(&self, _: A) -> io::Result<()> {
        self.0
    }
}

impl fmt::Debug for UdpSocket {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0
    }
}

#[derive(Debug)]
pub struct NonIpSockAddr {
    host: String,
}

impl error::Error for NonIpSockAddr {}

impl fmt::Display for NonIpSockAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Failed to convert address to SocketAddr: {}", self.host)
    }
}

pub struct LookupHost(!);

impl Iterator for LookupHost {
    type Item = SocketAddr;
    fn next(&mut self) -> Option<SocketAddr> {
        self.0
    }
}

pub fn lookup_host(host: &str, port: u16) -> io::Result<LookupHost> {
    Err(io::Error::new(
        io::ErrorKind::Uncategorized,
        NonIpSockAddr { host: format!("{host}:{port}") },
    ))
}
