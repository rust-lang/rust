#[cfg(test)]
mod tests;

use crate::cmp;
use crate::ffi::CString;
use crate::fmt;
use crate::io::{self, ErrorKind, IoSlice, IoSliceMut};
use crate::mem;
use crate::net::{Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr};
use crate::ptr;
use crate::sys::net::netc as c;
use crate::sys::net::{cvt, cvt_gai, cvt_r, init, wrlen_t, Socket};
use crate::sys_common::{AsInner, FromInner, IntoInner};
use crate::time::Duration;

use libc::{c_int, c_void};

cfg_if::cfg_if! {
    if #[cfg(any(
        target_os = "dragonfly", target_os = "freebsd",
        target_os = "ios", target_os = "macos", target_os = "watchos",
        target_os = "openbsd", target_os = "netbsd", target_os = "illumos",
        target_os = "solaris", target_os = "haiku", target_os = "l4re"))] {
        use crate::sys::net::netc::IPV6_JOIN_GROUP as IPV6_ADD_MEMBERSHIP;
        use crate::sys::net::netc::IPV6_LEAVE_GROUP as IPV6_DROP_MEMBERSHIP;
    } else {
        use crate::sys::net::netc::IPV6_ADD_MEMBERSHIP;
        use crate::sys::net::netc::IPV6_DROP_MEMBERSHIP;
    }
}

cfg_if::cfg_if! {
    if #[cfg(any(
        target_os = "linux", target_os = "android",
        target_os = "dragonfly", target_os = "freebsd",
        target_os = "openbsd", target_os = "netbsd",
        target_os = "haiku"))] {
        use libc::MSG_NOSIGNAL;
    } else {
        const MSG_NOSIGNAL: c_int = 0x0;
    }
}

cfg_if::cfg_if! {
    if #[cfg(any(
        target_os = "dragonfly", target_os = "freebsd",
        target_os = "openbsd", target_os = "netbsd",
        target_os = "solaris", target_os = "illumos"))] {
        use libc::c_uchar;
        type IpV4MultiCastType = c_uchar;
    } else {
        type IpV4MultiCastType = c_int;
    }
}

////////////////////////////////////////////////////////////////////////////////
// sockaddr and misc bindings
////////////////////////////////////////////////////////////////////////////////

pub fn setsockopt<T>(
    sock: &Socket,
    level: c_int,
    option_name: c_int,
    option_value: T,
) -> io::Result<()> {
    unsafe {
        cvt(c::setsockopt(
            sock.as_raw(),
            level,
            option_name,
            &option_value as *const T as *const _,
            mem::size_of::<T>() as c::socklen_t,
        ))?;
        Ok(())
    }
}

pub fn getsockopt<T: Copy>(sock: &Socket, level: c_int, option_name: c_int) -> io::Result<T> {
    unsafe {
        let mut option_value: T = mem::zeroed();
        let mut option_len = mem::size_of::<T>() as c::socklen_t;
        cvt(c::getsockopt(
            sock.as_raw(),
            level,
            option_name,
            &mut option_value as *mut T as *mut _,
            &mut option_len,
        ))?;
        Ok(option_value)
    }
}

fn sockname<F>(f: F) -> io::Result<SocketAddr>
where
    F: FnOnce(*mut c::sockaddr, *mut c::socklen_t) -> c_int,
{
    unsafe {
        let mut storage: c::sockaddr_storage = mem::zeroed();
        let mut len = mem::size_of_val(&storage) as c::socklen_t;
        cvt(f(&mut storage as *mut _ as *mut _, &mut len))?;
        sockaddr_to_addr(&storage, len as usize)
    }
}

pub fn sockaddr_to_addr(storage: &c::sockaddr_storage, len: usize) -> io::Result<SocketAddr> {
    match storage.ss_family as c_int {
        c::AF_INET => {
            assert!(len as usize >= mem::size_of::<c::sockaddr_in>());
            Ok(SocketAddr::V4(FromInner::from_inner(unsafe {
                *(storage as *const _ as *const c::sockaddr_in)
            })))
        }
        c::AF_INET6 => {
            assert!(len as usize >= mem::size_of::<c::sockaddr_in6>());
            Ok(SocketAddr::V6(FromInner::from_inner(unsafe {
                *(storage as *const _ as *const c::sockaddr_in6)
            })))
        }
        _ => Err(io::const_io_error!(ErrorKind::InvalidInput, "invalid argument")),
    }
}

#[cfg(target_os = "android")]
fn to_ipv6mr_interface(value: u32) -> c_int {
    value as c_int
}

#[cfg(not(target_os = "android"))]
fn to_ipv6mr_interface(value: u32) -> libc::c_uint {
    value as libc::c_uint
}

////////////////////////////////////////////////////////////////////////////////
// get_host_addresses
////////////////////////////////////////////////////////////////////////////////

pub struct LookupHost {
    original: *mut c::addrinfo,
    cur: *mut c::addrinfo,
    port: u16,
}

impl LookupHost {
    pub fn port(&self) -> u16 {
        self.port
    }
}

impl Iterator for LookupHost {
    type Item = SocketAddr;
    fn next(&mut self) -> Option<SocketAddr> {
        loop {
            unsafe {
                let cur = self.cur.as_ref()?;
                self.cur = cur.ai_next;
                match sockaddr_to_addr(mem::transmute(cur.ai_addr), cur.ai_addrlen as usize) {
                    Ok(addr) => return Some(addr),
                    Err(_) => continue,
                }
            }
        }
    }
}

unsafe impl Sync for LookupHost {}
unsafe impl Send for LookupHost {}

impl Drop for LookupHost {
    fn drop(&mut self) {
        unsafe { c::freeaddrinfo(self.original) }
    }
}

impl TryFrom<&str> for LookupHost {
    type Error = io::Error;

    fn try_from(s: &str) -> io::Result<LookupHost> {
        macro_rules! try_opt {
            ($e:expr, $msg:expr) => {
                match $e {
                    Some(r) => r,
                    None => return Err(io::const_io_error!(io::ErrorKind::InvalidInput, $msg)),
                }
            };
        }

        // split the string by ':' and convert the second part to u16
        let (host, port_str) = try_opt!(s.rsplit_once(':'), "invalid socket address");
        let port: u16 = try_opt!(port_str.parse().ok(), "invalid port value");
        (host, port).try_into()
    }
}

impl<'a> TryFrom<(&'a str, u16)> for LookupHost {
    type Error = io::Error;

    fn try_from((host, port): (&'a str, u16)) -> io::Result<LookupHost> {
        init();

        let c_host = CString::new(host)?;
        let mut hints: c::addrinfo = unsafe { mem::zeroed() };
        hints.ai_socktype = c::SOCK_STREAM;
        let mut res = ptr::null_mut();
        unsafe {
            cvt_gai(c::getaddrinfo(c_host.as_ptr(), ptr::null(), &hints, &mut res))
                .map(|_| LookupHost { original: res, cur: res, port })
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// TCP streams
////////////////////////////////////////////////////////////////////////////////

pub struct TcpStream {
    inner: Socket,
}

impl TcpStream {
    pub fn connect(addr: io::Result<&SocketAddr>) -> io::Result<TcpStream> {
        let addr = addr?;

        init();

        let sock = Socket::new(addr, c::SOCK_STREAM)?;

        let (addr, len) = addr.into_inner();
        cvt_r(|| unsafe { c::connect(sock.as_raw(), addr.as_ptr(), len) })?;
        Ok(TcpStream { inner: sock })
    }

    pub fn connect_timeout(addr: &SocketAddr, timeout: Duration) -> io::Result<TcpStream> {
        init();

        let sock = Socket::new(addr, c::SOCK_STREAM)?;
        sock.connect_timeout(addr, timeout)?;
        Ok(TcpStream { inner: sock })
    }

    pub fn socket(&self) -> &Socket {
        &self.inner
    }

    pub fn into_socket(self) -> Socket {
        self.inner
    }

    pub fn set_read_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        self.inner.set_timeout(dur, c::SO_RCVTIMEO)
    }

    pub fn set_write_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        self.inner.set_timeout(dur, c::SO_SNDTIMEO)
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        self.inner.timeout(c::SO_RCVTIMEO)
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        self.inner.timeout(c::SO_SNDTIMEO)
    }

    pub fn peek(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.peek(buf)
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.inner.read_vectored(bufs)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        self.inner.is_read_vectored()
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let len = cmp::min(buf.len(), <wrlen_t>::MAX as usize) as wrlen_t;
        let ret = cvt(unsafe {
            c::send(self.inner.as_raw(), buf.as_ptr() as *const c_void, len, MSG_NOSIGNAL)
        })?;
        Ok(ret as usize)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.inner.write_vectored(bufs)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        sockname(|buf, len| unsafe { c::getpeername(self.inner.as_raw(), buf, len) })
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        sockname(|buf, len| unsafe { c::getsockname(self.inner.as_raw(), buf, len) })
    }

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        self.inner.shutdown(how)
    }

    pub fn duplicate(&self) -> io::Result<TcpStream> {
        self.inner.duplicate().map(|s| TcpStream { inner: s })
    }

    pub fn set_linger(&self, linger: Option<Duration>) -> io::Result<()> {
        self.inner.set_linger(linger)
    }

    pub fn linger(&self) -> io::Result<Option<Duration>> {
        self.inner.linger()
    }

    pub fn set_nodelay(&self, nodelay: bool) -> io::Result<()> {
        self.inner.set_nodelay(nodelay)
    }

    pub fn nodelay(&self) -> io::Result<bool> {
        self.inner.nodelay()
    }

    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        setsockopt(&self.inner, c::IPPROTO_IP, c::IP_TTL, ttl as c_int)
    }

    pub fn ttl(&self) -> io::Result<u32> {
        let raw: c_int = getsockopt(&self.inner, c::IPPROTO_IP, c::IP_TTL)?;
        Ok(raw as u32)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        self.inner.take_error()
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.inner.set_nonblocking(nonblocking)
    }
}

impl AsInner<Socket> for TcpStream {
    fn as_inner(&self) -> &Socket {
        &self.inner
    }
}

impl FromInner<Socket> for TcpStream {
    fn from_inner(socket: Socket) -> TcpStream {
        TcpStream { inner: socket }
    }
}

impl fmt::Debug for TcpStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut res = f.debug_struct("TcpStream");

        if let Ok(addr) = self.socket_addr() {
            res.field("addr", &addr);
        }

        if let Ok(peer) = self.peer_addr() {
            res.field("peer", &peer);
        }

        let name = if cfg!(windows) { "socket" } else { "fd" };
        res.field(name, &self.inner.as_raw()).finish()
    }
}

////////////////////////////////////////////////////////////////////////////////
// TCP listeners
////////////////////////////////////////////////////////////////////////////////

pub struct TcpListener {
    inner: Socket,
}

impl TcpListener {
    pub fn bind(addr: io::Result<&SocketAddr>) -> io::Result<TcpListener> {
        let addr = addr?;

        init();

        let sock = Socket::new(addr, c::SOCK_STREAM)?;

        // On platforms with Berkeley-derived sockets, this allows to quickly
        // rebind a socket, without needing to wait for the OS to clean up the
        // previous one.
        //
        // On Windows, this allows rebinding sockets which are actively in use,
        // which allows “socket hijacking”, so we explicitly don't set it here.
        // https://docs.microsoft.com/en-us/windows/win32/winsock/using-so-reuseaddr-and-so-exclusiveaddruse
        #[cfg(not(windows))]
        setsockopt(&sock, c::SOL_SOCKET, c::SO_REUSEADDR, 1 as c_int)?;

        // Bind our new socket
        let (addr, len) = addr.into_inner();
        cvt(unsafe { c::bind(sock.as_raw(), addr.as_ptr(), len as _) })?;

        cfg_if::cfg_if! {
            if #[cfg(target_os = "horizon")] {
                // The 3DS doesn't support a big connection backlog. Sometimes
                // it allows up to about 37, but other times it doesn't even
                // accept 32. There may be a global limitation causing this.
                let backlog = 20;
            } else {
                // The default for all other platforms
                let backlog = 128;
            }
        }

        // Start listening
        cvt(unsafe { c::listen(sock.as_raw(), backlog) })?;
        Ok(TcpListener { inner: sock })
    }

    pub fn socket(&self) -> &Socket {
        &self.inner
    }

    pub fn into_socket(self) -> Socket {
        self.inner
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        sockname(|buf, len| unsafe { c::getsockname(self.inner.as_raw(), buf, len) })
    }

    pub fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        let mut storage: c::sockaddr_storage = unsafe { mem::zeroed() };
        let mut len = mem::size_of_val(&storage) as c::socklen_t;
        let sock = self.inner.accept(&mut storage as *mut _ as *mut _, &mut len)?;
        let addr = sockaddr_to_addr(&storage, len as usize)?;
        Ok((TcpStream { inner: sock }, addr))
    }

    pub fn duplicate(&self) -> io::Result<TcpListener> {
        self.inner.duplicate().map(|s| TcpListener { inner: s })
    }

    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        setsockopt(&self.inner, c::IPPROTO_IP, c::IP_TTL, ttl as c_int)
    }

    pub fn ttl(&self) -> io::Result<u32> {
        let raw: c_int = getsockopt(&self.inner, c::IPPROTO_IP, c::IP_TTL)?;
        Ok(raw as u32)
    }

    pub fn set_only_v6(&self, only_v6: bool) -> io::Result<()> {
        setsockopt(&self.inner, c::IPPROTO_IPV6, c::IPV6_V6ONLY, only_v6 as c_int)
    }

    pub fn only_v6(&self) -> io::Result<bool> {
        let raw: c_int = getsockopt(&self.inner, c::IPPROTO_IPV6, c::IPV6_V6ONLY)?;
        Ok(raw != 0)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        self.inner.take_error()
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.inner.set_nonblocking(nonblocking)
    }
}

impl FromInner<Socket> for TcpListener {
    fn from_inner(socket: Socket) -> TcpListener {
        TcpListener { inner: socket }
    }
}

impl fmt::Debug for TcpListener {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut res = f.debug_struct("TcpListener");

        if let Ok(addr) = self.socket_addr() {
            res.field("addr", &addr);
        }

        let name = if cfg!(windows) { "socket" } else { "fd" };
        res.field(name, &self.inner.as_raw()).finish()
    }
}

////////////////////////////////////////////////////////////////////////////////
// UDP
////////////////////////////////////////////////////////////////////////////////

pub struct UdpSocket {
    inner: Socket,
}

impl UdpSocket {
    pub fn bind(addr: io::Result<&SocketAddr>) -> io::Result<UdpSocket> {
        let addr = addr?;

        init();

        let sock = Socket::new(addr, c::SOCK_DGRAM)?;
        let (addr, len) = addr.into_inner();
        cvt(unsafe { c::bind(sock.as_raw(), addr.as_ptr(), len as _) })?;
        Ok(UdpSocket { inner: sock })
    }

    pub fn socket(&self) -> &Socket {
        &self.inner
    }

    pub fn into_socket(self) -> Socket {
        self.inner
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        sockname(|buf, len| unsafe { c::getpeername(self.inner.as_raw(), buf, len) })
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        sockname(|buf, len| unsafe { c::getsockname(self.inner.as_raw(), buf, len) })
    }

    pub fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.inner.recv_from(buf)
    }

    pub fn peek_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.inner.peek_from(buf)
    }

    pub fn send_to(&self, buf: &[u8], dst: &SocketAddr) -> io::Result<usize> {
        let len = cmp::min(buf.len(), <wrlen_t>::MAX as usize) as wrlen_t;
        let (dst, dstlen) = dst.into_inner();
        let ret = cvt(unsafe {
            c::sendto(
                self.inner.as_raw(),
                buf.as_ptr() as *const c_void,
                len,
                MSG_NOSIGNAL,
                dst.as_ptr(),
                dstlen,
            )
        })?;
        Ok(ret as usize)
    }

    pub fn duplicate(&self) -> io::Result<UdpSocket> {
        self.inner.duplicate().map(|s| UdpSocket { inner: s })
    }

    pub fn set_read_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        self.inner.set_timeout(dur, c::SO_RCVTIMEO)
    }

    pub fn set_write_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        self.inner.set_timeout(dur, c::SO_SNDTIMEO)
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        self.inner.timeout(c::SO_RCVTIMEO)
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        self.inner.timeout(c::SO_SNDTIMEO)
    }

    pub fn set_broadcast(&self, broadcast: bool) -> io::Result<()> {
        setsockopt(&self.inner, c::SOL_SOCKET, c::SO_BROADCAST, broadcast as c_int)
    }

    pub fn broadcast(&self) -> io::Result<bool> {
        let raw: c_int = getsockopt(&self.inner, c::SOL_SOCKET, c::SO_BROADCAST)?;
        Ok(raw != 0)
    }

    pub fn set_multicast_loop_v4(&self, multicast_loop_v4: bool) -> io::Result<()> {
        setsockopt(
            &self.inner,
            c::IPPROTO_IP,
            c::IP_MULTICAST_LOOP,
            multicast_loop_v4 as IpV4MultiCastType,
        )
    }

    pub fn multicast_loop_v4(&self) -> io::Result<bool> {
        let raw: IpV4MultiCastType = getsockopt(&self.inner, c::IPPROTO_IP, c::IP_MULTICAST_LOOP)?;
        Ok(raw != 0)
    }

    pub fn set_multicast_ttl_v4(&self, multicast_ttl_v4: u32) -> io::Result<()> {
        setsockopt(
            &self.inner,
            c::IPPROTO_IP,
            c::IP_MULTICAST_TTL,
            multicast_ttl_v4 as IpV4MultiCastType,
        )
    }

    pub fn multicast_ttl_v4(&self) -> io::Result<u32> {
        let raw: IpV4MultiCastType = getsockopt(&self.inner, c::IPPROTO_IP, c::IP_MULTICAST_TTL)?;
        Ok(raw as u32)
    }

    pub fn set_multicast_loop_v6(&self, multicast_loop_v6: bool) -> io::Result<()> {
        setsockopt(&self.inner, c::IPPROTO_IPV6, c::IPV6_MULTICAST_LOOP, multicast_loop_v6 as c_int)
    }

    pub fn multicast_loop_v6(&self) -> io::Result<bool> {
        let raw: c_int = getsockopt(&self.inner, c::IPPROTO_IPV6, c::IPV6_MULTICAST_LOOP)?;
        Ok(raw != 0)
    }

    pub fn join_multicast_v4(&self, multiaddr: &Ipv4Addr, interface: &Ipv4Addr) -> io::Result<()> {
        let mreq = c::ip_mreq {
            imr_multiaddr: multiaddr.into_inner(),
            imr_interface: interface.into_inner(),
        };
        setsockopt(&self.inner, c::IPPROTO_IP, c::IP_ADD_MEMBERSHIP, mreq)
    }

    pub fn join_multicast_v6(&self, multiaddr: &Ipv6Addr, interface: u32) -> io::Result<()> {
        let mreq = c::ipv6_mreq {
            ipv6mr_multiaddr: multiaddr.into_inner(),
            ipv6mr_interface: to_ipv6mr_interface(interface),
        };
        setsockopt(&self.inner, c::IPPROTO_IPV6, IPV6_ADD_MEMBERSHIP, mreq)
    }

    pub fn leave_multicast_v4(&self, multiaddr: &Ipv4Addr, interface: &Ipv4Addr) -> io::Result<()> {
        let mreq = c::ip_mreq {
            imr_multiaddr: multiaddr.into_inner(),
            imr_interface: interface.into_inner(),
        };
        setsockopt(&self.inner, c::IPPROTO_IP, c::IP_DROP_MEMBERSHIP, mreq)
    }

    pub fn leave_multicast_v6(&self, multiaddr: &Ipv6Addr, interface: u32) -> io::Result<()> {
        let mreq = c::ipv6_mreq {
            ipv6mr_multiaddr: multiaddr.into_inner(),
            ipv6mr_interface: to_ipv6mr_interface(interface),
        };
        setsockopt(&self.inner, c::IPPROTO_IPV6, IPV6_DROP_MEMBERSHIP, mreq)
    }

    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        setsockopt(&self.inner, c::IPPROTO_IP, c::IP_TTL, ttl as c_int)
    }

    pub fn ttl(&self) -> io::Result<u32> {
        let raw: c_int = getsockopt(&self.inner, c::IPPROTO_IP, c::IP_TTL)?;
        Ok(raw as u32)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        self.inner.take_error()
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.inner.set_nonblocking(nonblocking)
    }

    pub fn recv(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }

    pub fn peek(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.peek(buf)
    }

    pub fn send(&self, buf: &[u8]) -> io::Result<usize> {
        let len = cmp::min(buf.len(), <wrlen_t>::MAX as usize) as wrlen_t;
        let ret = cvt(unsafe {
            c::send(self.inner.as_raw(), buf.as_ptr() as *const c_void, len, MSG_NOSIGNAL)
        })?;
        Ok(ret as usize)
    }

    pub fn connect(&self, addr: io::Result<&SocketAddr>) -> io::Result<()> {
        let (addr, len) = addr?.into_inner();
        cvt_r(|| unsafe { c::connect(self.inner.as_raw(), addr.as_ptr(), len) }).map(drop)
    }
}

impl FromInner<Socket> for UdpSocket {
    fn from_inner(socket: Socket) -> UdpSocket {
        UdpSocket { inner: socket }
    }
}

impl fmt::Debug for UdpSocket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut res = f.debug_struct("UdpSocket");

        if let Ok(addr) = self.socket_addr() {
            res.field("addr", &addr);
        }

        let name = if cfg!(windows) { "socket" } else { "fd" };
        res.field(name, &self.inner.as_raw()).finish()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Converting SocketAddr to libc representation
////////////////////////////////////////////////////////////////////////////////

/// A type with the same memory layout as `c::sockaddr`. Used in converting Rust level
/// SocketAddr* types into their system representation. The benefit of this specific
/// type over using `c::sockaddr_storage` is that this type is exactly as large as it
/// needs to be and not a lot larger. And it can be initialized more cleanly from Rust.
#[repr(C)]
pub(crate) union SocketAddrCRepr {
    v4: c::sockaddr_in,
    v6: c::sockaddr_in6,
}

impl SocketAddrCRepr {
    pub fn as_ptr(&self) -> *const c::sockaddr {
        self as *const _ as *const c::sockaddr
    }
}

impl<'a> IntoInner<(SocketAddrCRepr, c::socklen_t)> for &'a SocketAddr {
    fn into_inner(self) -> (SocketAddrCRepr, c::socklen_t) {
        match *self {
            SocketAddr::V4(ref a) => {
                let sockaddr = SocketAddrCRepr { v4: a.into_inner() };
                (sockaddr, mem::size_of::<c::sockaddr_in>() as c::socklen_t)
            }
            SocketAddr::V6(ref a) => {
                let sockaddr = SocketAddrCRepr { v6: a.into_inner() };
                (sockaddr, mem::size_of::<c::sockaddr_in6>() as c::socklen_t)
            }
        }
    }
}
