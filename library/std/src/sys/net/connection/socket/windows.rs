#![unstable(issue = "none", feature = "windows_net")]

use core::ffi::{c_int, c_long, c_ulong, c_ushort};

use super::{getsockopt, setsockopt, socket_addr_from_c, socket_addr_to_c};
use crate::io::{self, BorrowedBuf, BorrowedCursor, IoSlice, IoSliceMut, Read};
use crate::net::{Shutdown, SocketAddr};
use crate::os::windows::io::{
    AsRawSocket, AsSocket, BorrowedSocket, FromRawSocket, IntoRawSocket, OwnedSocket, RawSocket,
};
use crate::sys::c;
use crate::sys::pal::winsock::last_error;
use crate::sys_common::{AsInner, FromInner, IntoInner};
use crate::time::Duration;
use crate::{cmp, mem, ptr, sys};

#[allow(non_camel_case_types)]
pub type wrlen_t = i32;

pub(super) mod netc {
    //! BSD socket compatibility shim
    //!
    //! Some Windows API types are not quite what's expected by our cross-platform
    //! net code. E.g. naming differences or different pointer types.

    use core::ffi::{c_char, c_int, c_uint, c_ulong, c_ushort, c_void};

    use crate::sys::c::{self, ADDRESS_FAMILY, ADDRINFOA, SOCKADDR, SOCKET};
    // re-exports from Windows API bindings.
    pub use crate::sys::c::{
        ADDRESS_FAMILY as sa_family_t, ADDRINFOA as addrinfo, IP_ADD_MEMBERSHIP,
        IP_DROP_MEMBERSHIP, IP_MULTICAST_LOOP, IP_MULTICAST_TTL, IP_TTL, IPPROTO_IP, IPPROTO_IPV6,
        IPV6_ADD_MEMBERSHIP, IPV6_DROP_MEMBERSHIP, IPV6_MULTICAST_LOOP, IPV6_V6ONLY, SO_BROADCAST,
        SO_RCVTIMEO, SO_SNDTIMEO, SOCK_DGRAM, SOCK_STREAM, SOCKADDR as sockaddr,
        SOCKADDR_STORAGE as sockaddr_storage, SOL_SOCKET, bind, connect, freeaddrinfo, getpeername,
        getsockname, getsockopt, listen, setsockopt,
    };

    #[allow(non_camel_case_types)]
    pub type socklen_t = c_int;

    pub const AF_INET: i32 = c::AF_INET as i32;
    pub const AF_INET6: i32 = c::AF_INET6 as i32;

    // The following two structs use a union in the generated bindings but
    // our cross-platform code expects a normal field so it's redefined here.
    // As a consequence, we also need to redefine other structs that use this struct.
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct in_addr {
        pub s_addr: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct in6_addr {
        pub s6_addr: [u8; 16],
    }

    #[repr(C)]
    pub struct ip_mreq {
        pub imr_multiaddr: in_addr,
        pub imr_interface: in_addr,
    }

    #[repr(C)]
    pub struct ipv6_mreq {
        pub ipv6mr_multiaddr: in6_addr,
        pub ipv6mr_interface: c_uint,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct sockaddr_in {
        pub sin_family: ADDRESS_FAMILY,
        pub sin_port: c_ushort,
        pub sin_addr: in_addr,
        pub sin_zero: [c_char; 8],
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct sockaddr_in6 {
        pub sin6_family: ADDRESS_FAMILY,
        pub sin6_port: c_ushort,
        pub sin6_flowinfo: c_ulong,
        pub sin6_addr: in6_addr,
        pub sin6_scope_id: c_ulong,
    }

    pub unsafe fn send(socket: SOCKET, buf: *const c_void, len: c_int, flags: c_int) -> c_int {
        unsafe { c::send(socket, buf.cast::<u8>(), len, flags) }
    }
    pub unsafe fn sendto(
        socket: SOCKET,
        buf: *const c_void,
        len: c_int,
        flags: c_int,
        addr: *const SOCKADDR,
        addrlen: c_int,
    ) -> c_int {
        unsafe { c::sendto(socket, buf.cast::<u8>(), len, flags, addr, addrlen) }
    }
    pub unsafe fn getaddrinfo(
        node: *const c_char,
        service: *const c_char,
        hints: *const ADDRINFOA,
        res: *mut *mut ADDRINFOA,
    ) -> c_int {
        unsafe { c::getaddrinfo(node.cast::<u8>(), service.cast::<u8>(), hints, res) }
    }
}

pub use crate::sys::pal::winsock::{cvt, cvt_gai, cvt_r, startup as init};

#[expect(missing_debug_implementations)]
pub struct Socket(OwnedSocket);

impl Socket {
    pub fn new(family: c_int, ty: c_int) -> io::Result<Socket> {
        let socket = unsafe {
            c::WSASocketW(
                family,
                ty,
                0,
                ptr::null_mut(),
                0,
                c::WSA_FLAG_OVERLAPPED | c::WSA_FLAG_NO_HANDLE_INHERIT,
            )
        };

        if socket != c::INVALID_SOCKET {
            unsafe { Ok(Self::from_raw(socket)) }
        } else {
            let error = unsafe { c::WSAGetLastError() };

            if error != c::WSAEPROTOTYPE && error != c::WSAEINVAL {
                return Err(io::Error::from_raw_os_error(error));
            }

            let socket =
                unsafe { c::WSASocketW(family, ty, 0, ptr::null_mut(), 0, c::WSA_FLAG_OVERLAPPED) };

            if socket == c::INVALID_SOCKET {
                return Err(last_error());
            }

            unsafe {
                let socket = Self::from_raw(socket);
                socket.0.set_no_inherit()?;
                Ok(socket)
            }
        }
    }

    pub fn connect(&self, addr: &SocketAddr) -> io::Result<()> {
        let (addr, len) = socket_addr_to_c(addr);
        let result = unsafe { c::connect(self.as_raw(), addr.as_ptr(), len) };
        cvt(result).map(drop)
    }

    pub fn connect_timeout(&self, addr: &SocketAddr, timeout: Duration) -> io::Result<()> {
        self.set_nonblocking(true)?;
        let result = self.connect(addr);
        self.set_nonblocking(false)?;

        match result {
            Err(ref error) if error.kind() == io::ErrorKind::WouldBlock => {
                if timeout.as_secs() == 0 && timeout.subsec_nanos() == 0 {
                    return Err(io::Error::ZERO_TIMEOUT);
                }

                let mut timeout = c::TIMEVAL {
                    tv_sec: cmp::min(timeout.as_secs(), c_long::MAX as u64) as c_long,
                    tv_usec: timeout.subsec_micros() as c_long,
                };

                if timeout.tv_sec == 0 && timeout.tv_usec == 0 {
                    timeout.tv_usec = 1;
                }

                let fds = {
                    let mut fds = unsafe { mem::zeroed::<c::FD_SET>() };
                    fds.fd_count = 1;
                    fds.fd_array[0] = self.as_raw();
                    fds
                };

                let mut writefds = fds;
                let mut errorfds = fds;

                let count = {
                    let result = unsafe {
                        c::select(1, ptr::null_mut(), &mut writefds, &mut errorfds, &timeout)
                    };
                    cvt(result)?
                };

                match count {
                    0 => Err(io::const_error!(io::ErrorKind::TimedOut, "connection timed out")),
                    _ => {
                        if writefds.fd_count != 1 {
                            if let Some(e) = self.take_error()? {
                                return Err(e);
                            }
                        }

                        Ok(())
                    }
                }
            }
            _ => result,
        }
    }

    pub fn accept(&self, storage: *mut c::SOCKADDR, len: *mut c_int) -> io::Result<Socket> {
        let socket = unsafe { c::accept(self.as_raw(), storage, len) };

        match socket {
            c::INVALID_SOCKET => Err(last_error()),
            _ => unsafe { Ok(Self::from_raw(socket)) },
        }
    }

    pub fn duplicate(&self) -> io::Result<Socket> {
        Ok(Self(self.0.try_clone()?))
    }

    fn recv_with_flags(&self, mut buf: BorrowedCursor<'_>, flags: c_int) -> io::Result<()> {
        // On unix when a socket is shut down all further reads return 0, so we
        // do the same on windows to map a shut down socket to returning EOF.
        let length = cmp::min(buf.capacity(), i32::MAX as usize) as i32;
        let result =
            unsafe { c::recv(self.as_raw(), buf.as_mut().as_mut_ptr() as *mut _, length, flags) };

        match result {
            c::SOCKET_ERROR => {
                let error = unsafe { c::WSAGetLastError() };

                if error == c::WSAESHUTDOWN {
                    Ok(())
                } else {
                    Err(io::Error::from_raw_os_error(error))
                }
            }
            _ => {
                unsafe { buf.advance_unchecked(result as usize) };
                Ok(())
            }
        }
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let mut buf = BorrowedBuf::from(buf);
        self.recv_with_flags(buf.unfilled(), 0)?;
        Ok(buf.len())
    }

    pub fn read_buf(&self, buf: BorrowedCursor<'_>) -> io::Result<()> {
        self.recv_with_flags(buf, 0)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        // On unix when a socket is shut down all further reads return 0, so we
        // do the same on windows to map a shut down socket to returning EOF.
        let length = cmp::min(bufs.len(), u32::MAX as usize) as u32;
        let mut nread = 0;
        let mut flags = 0;
        let result = unsafe {
            c::WSARecv(
                self.as_raw(),
                bufs.as_mut_ptr() as *mut c::WSABUF,
                length,
                &mut nread,
                &mut flags,
                ptr::null_mut(),
                None,
            )
        };

        match result {
            0 => Ok(nread as usize),
            _ => {
                let error = unsafe { c::WSAGetLastError() };

                if error == c::WSAESHUTDOWN {
                    Ok(0)
                } else {
                    Err(io::Error::from_raw_os_error(error))
                }
            }
        }
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        true
    }

    pub fn peek(&self, buf: &mut [u8]) -> io::Result<usize> {
        let mut buf = BorrowedBuf::from(buf);
        self.recv_with_flags(buf.unfilled(), c::MSG_PEEK)?;
        Ok(buf.len())
    }

    fn recv_from_with_flags(
        &self,
        buf: &mut [u8],
        flags: c_int,
    ) -> io::Result<(usize, SocketAddr)> {
        let mut storage = unsafe { mem::zeroed::<c::SOCKADDR_STORAGE>() };
        let mut addrlen = size_of_val(&storage) as netc::socklen_t;
        let length = cmp::min(buf.len(), <wrlen_t>::MAX as usize) as wrlen_t;

        // On unix when a socket is shut down all further reads return 0, so we
        // do the same on windows to map a shut down socket to returning EOF.
        let result = unsafe {
            c::recvfrom(
                self.as_raw(),
                buf.as_mut_ptr() as *mut _,
                length,
                flags,
                (&raw mut storage) as *mut _,
                &mut addrlen,
            )
        };

        match result {
            c::SOCKET_ERROR => {
                let error = unsafe { c::WSAGetLastError() };

                if error == c::WSAESHUTDOWN {
                    Ok((0, unsafe { socket_addr_from_c(&storage, addrlen as usize)? }))
                } else {
                    Err(io::Error::from_raw_os_error(error))
                }
            }
            _ => Ok((result as usize, unsafe { socket_addr_from_c(&storage, addrlen as usize)? })),
        }
    }

    pub fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.recv_from_with_flags(buf, 0)
    }

    pub fn peek_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.recv_from_with_flags(buf, c::MSG_PEEK)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let length = cmp::min(bufs.len(), u32::MAX as usize) as u32;
        let mut nwritten = 0;
        let result = unsafe {
            c::WSASend(
                self.as_raw(),
                bufs.as_ptr() as *const c::WSABUF as *mut _,
                length,
                &mut nwritten,
                0,
                ptr::null_mut(),
                None,
            )
        };
        cvt(result).map(|_| nwritten as usize)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        true
    }

    pub fn set_timeout(&self, dur: Option<Duration>, kind: c_int) -> io::Result<()> {
        let timeout = match dur {
            Some(dur) => {
                let timeout = sys::dur2timeout(dur);
                if timeout == 0 {
                    return Err(io::Error::ZERO_TIMEOUT);
                }
                timeout
            }
            None => 0,
        };
        unsafe { setsockopt(self, c::SOL_SOCKET, kind, timeout) }
    }

    pub fn timeout(&self, kind: c_int) -> io::Result<Option<Duration>> {
        let raw: u32 = unsafe { getsockopt(self, c::SOL_SOCKET, kind)? };
        if raw == 0 {
            Ok(None)
        } else {
            let secs = raw / 1000;
            let nsec = (raw % 1000) * 1000000;
            Ok(Some(Duration::new(secs as u64, nsec as u32)))
        }
    }

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        let how = match how {
            Shutdown::Write => c::SD_SEND,
            Shutdown::Read => c::SD_RECEIVE,
            Shutdown::Both => c::SD_BOTH,
        };
        let result = unsafe { c::shutdown(self.as_raw(), how) };
        cvt(result).map(drop)
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        let mut nonblocking = nonblocking as c_ulong;
        let result =
            unsafe { c::ioctlsocket(self.as_raw(), c::FIONBIO as c_int, &mut nonblocking) };
        cvt(result).map(drop)
    }

    pub fn set_linger(&self, linger: Option<Duration>) -> io::Result<()> {
        let linger = c::LINGER {
            l_onoff: linger.is_some() as c_ushort,
            l_linger: linger.unwrap_or_default().as_secs() as c_ushort,
        };

        unsafe { setsockopt(self, c::SOL_SOCKET, c::SO_LINGER, linger) }
    }

    pub fn linger(&self) -> io::Result<Option<Duration>> {
        let val: c::LINGER = unsafe { getsockopt(self, c::SOL_SOCKET, c::SO_LINGER)? };

        Ok((val.l_onoff != 0).then(|| Duration::from_secs(val.l_linger as u64)))
    }

    pub fn set_nodelay(&self, nodelay: bool) -> io::Result<()> {
        unsafe { setsockopt(self, c::IPPROTO_TCP, c::TCP_NODELAY, nodelay as c::BOOL) }
    }

    pub fn nodelay(&self) -> io::Result<bool> {
        let raw: c::BOOL = unsafe { getsockopt(self, c::IPPROTO_TCP, c::TCP_NODELAY)? };
        Ok(raw != 0)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        let raw: c_int = unsafe { getsockopt(self, c::SOL_SOCKET, c::SO_ERROR)? };
        if raw == 0 { Ok(None) } else { Ok(Some(io::Error::from_raw_os_error(raw as i32))) }
    }

    // This is used by sys_common code to abstract over Windows and Unix.
    pub fn as_raw(&self) -> c::SOCKET {
        debug_assert_eq!(size_of::<c::SOCKET>(), size_of::<RawSocket>());
        debug_assert_eq!(align_of::<c::SOCKET>(), align_of::<RawSocket>());
        self.as_inner().as_raw_socket() as c::SOCKET
    }
    pub unsafe fn from_raw(raw: c::SOCKET) -> Self {
        debug_assert_eq!(size_of::<c::SOCKET>(), size_of::<RawSocket>());
        debug_assert_eq!(align_of::<c::SOCKET>(), align_of::<RawSocket>());
        unsafe { Self::from_raw_socket(raw as RawSocket) }
    }
}

#[unstable(reason = "not public", issue = "none", feature = "fd_read")]
impl<'a> Read for &'a Socket {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        (**self).read(buf)
    }
}

impl AsInner<OwnedSocket> for Socket {
    #[inline]
    fn as_inner(&self) -> &OwnedSocket {
        &self.0
    }
}

impl FromInner<OwnedSocket> for Socket {
    fn from_inner(sock: OwnedSocket) -> Socket {
        Socket(sock)
    }
}

impl IntoInner<OwnedSocket> for Socket {
    fn into_inner(self) -> OwnedSocket {
        self.0
    }
}

impl AsSocket for Socket {
    fn as_socket(&self) -> BorrowedSocket<'_> {
        self.0.as_socket()
    }
}

impl AsRawSocket for Socket {
    fn as_raw_socket(&self) -> RawSocket {
        self.0.as_raw_socket()
    }
}

impl IntoRawSocket for Socket {
    fn into_raw_socket(self) -> RawSocket {
        self.0.into_raw_socket()
    }
}

impl FromRawSocket for Socket {
    unsafe fn from_raw_socket(raw_socket: RawSocket) -> Self {
        unsafe { Self(FromRawSocket::from_raw_socket(raw_socket)) }
    }
}
