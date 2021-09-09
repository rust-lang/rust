#![unstable(issue = "none", feature = "windows_net")]

use crate::cmp;
use crate::io::{self, IoSlice, IoSliceMut, Read};
use crate::mem;
use crate::net::{Shutdown, SocketAddr};
use crate::os::windows::io::{
    AsRawSocket, AsSocket, BorrowedSocket, FromRawSocket, IntoRawSocket, OwnedSocket, RawSocket,
};
use crate::ptr;
use crate::sync::Once;
use crate::sys;
use crate::sys::c;
use crate::sys_common::net;
use crate::sys_common::{AsInner, FromInner, IntoInner};
use crate::time::Duration;

use libc::{c_int, c_long, c_ulong, c_ushort};

pub type wrlen_t = i32;

pub mod netc {
    pub use crate::sys::c::ADDRESS_FAMILY as sa_family_t;
    pub use crate::sys::c::ADDRINFOA as addrinfo;
    pub use crate::sys::c::SOCKADDR as sockaddr;
    pub use crate::sys::c::SOCKADDR_STORAGE_LH as sockaddr_storage;
    pub use crate::sys::c::*;
}

pub struct Socket(OwnedSocket);

static INIT: Once = Once::new();

/// Checks whether the Windows socket interface has been started already, and
/// if not, starts it.
pub fn init() {
    INIT.call_once(|| unsafe {
        let mut data: c::WSADATA = mem::zeroed();
        let ret = c::WSAStartup(
            0x202, // version 2.2
            &mut data,
        );
        assert_eq!(ret, 0);
    });
}

pub fn cleanup() {
    if INIT.is_completed() {
        // only close the socket interface if it has actually been started
        unsafe {
            c::WSACleanup();
        }
    }
}

/// Returns the last error from the Windows socket interface.
fn last_error() -> io::Error {
    io::Error::from_raw_os_error(unsafe { c::WSAGetLastError() })
}

#[doc(hidden)]
pub trait IsMinusOne {
    fn is_minus_one(&self) -> bool;
}

macro_rules! impl_is_minus_one {
    ($($t:ident)*) => ($(impl IsMinusOne for $t {
        fn is_minus_one(&self) -> bool {
            *self == -1
        }
    })*)
}

impl_is_minus_one! { i8 i16 i32 i64 isize }

/// Checks if the signed integer is the Windows constant `SOCKET_ERROR` (-1)
/// and if so, returns the last error from the Windows socket interface. This
/// function must be called before another call to the socket API is made.
pub fn cvt<T: IsMinusOne>(t: T) -> io::Result<T> {
    if t.is_minus_one() { Err(last_error()) } else { Ok(t) }
}

/// A variant of `cvt` for `getaddrinfo` which return 0 for a success.
pub fn cvt_gai(err: c_int) -> io::Result<()> {
    if err == 0 { Ok(()) } else { Err(last_error()) }
}

/// Just to provide the same interface as sys/unix/net.rs
pub fn cvt_r<T, F>(mut f: F) -> io::Result<T>
where
    T: IsMinusOne,
    F: FnMut() -> T,
{
    cvt(f())
}

impl Socket {
    pub fn new(addr: &SocketAddr, ty: c_int) -> io::Result<Socket> {
        let family = match *addr {
            SocketAddr::V4(..) => c::AF_INET,
            SocketAddr::V6(..) => c::AF_INET6,
        };
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
            unsafe { Ok(Self::from_raw_socket(socket)) }
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
                let socket = Self::from_raw_socket(socket);
                socket.set_no_inherit()?;
                Ok(socket)
            }
        }
    }

    pub fn connect_timeout(&self, addr: &SocketAddr, timeout: Duration) -> io::Result<()> {
        self.set_nonblocking(true)?;
        let result = {
            let (addrp, len) = addr.into_inner();
            let result = unsafe { c::connect(self.as_raw_socket(), addrp, len) };
            cvt(result).map(drop)
        };
        self.set_nonblocking(false)?;

        match result {
            Err(ref error) if error.kind() == io::ErrorKind::WouldBlock => {
                if timeout.as_secs() == 0 && timeout.subsec_nanos() == 0 {
                    return Err(io::Error::new_const(
                        io::ErrorKind::InvalidInput,
                        &"cannot set a 0 duration timeout",
                    ));
                }

                let mut timeout = c::timeval {
                    tv_sec: timeout.as_secs() as c_long,
                    tv_usec: (timeout.subsec_nanos() / 1000) as c_long,
                };

                if timeout.tv_sec == 0 && timeout.tv_usec == 0 {
                    timeout.tv_usec = 1;
                }

                let fds = {
                    let mut fds = unsafe { mem::zeroed::<c::fd_set>() };
                    fds.fd_count = 1;
                    fds.fd_array[0] = self.as_raw_socket();
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
                    0 => {
                        Err(io::Error::new_const(io::ErrorKind::TimedOut, &"connection timed out"))
                    }
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
        let socket = unsafe { c::accept(self.as_raw_socket(), storage, len) };

        match socket {
            c::INVALID_SOCKET => Err(last_error()),
            _ => unsafe { Ok(Self::from_raw_socket(socket)) },
        }
    }

    pub fn duplicate(&self) -> io::Result<Socket> {
        Ok(Self(self.0.duplicate()?))
    }

    fn recv_with_flags(&self, buf: &mut [u8], flags: c_int) -> io::Result<usize> {
        // On unix when a socket is shut down all further reads return 0, so we
        // do the same on windows to map a shut down socket to returning EOF.
        let length = cmp::min(buf.len(), i32::MAX as usize) as i32;
        let result =
            unsafe { c::recv(self.as_raw_socket(), buf.as_mut_ptr() as *mut _, length, flags) };

        match result {
            c::SOCKET_ERROR => {
                let error = unsafe { c::WSAGetLastError() };

                if error == c::WSAESHUTDOWN {
                    Ok(0)
                } else {
                    Err(io::Error::from_raw_os_error(error))
                }
            }
            _ => Ok(result as usize),
        }
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.recv_with_flags(buf, 0)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        // On unix when a socket is shut down all further reads return 0, so we
        // do the same on windows to map a shut down socket to returning EOF.
        let length = cmp::min(bufs.len(), c::DWORD::MAX as usize) as c::DWORD;
        let mut nread = 0;
        let mut flags = 0;
        let result = unsafe {
            c::WSARecv(
                self.as_raw_socket(),
                bufs.as_mut_ptr() as *mut c::WSABUF,
                length,
                &mut nread,
                &mut flags,
                ptr::null_mut(),
                ptr::null_mut(),
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
        self.recv_with_flags(buf, c::MSG_PEEK)
    }

    fn recv_from_with_flags(
        &self,
        buf: &mut [u8],
        flags: c_int,
    ) -> io::Result<(usize, SocketAddr)> {
        let mut storage = unsafe { mem::zeroed::<c::SOCKADDR_STORAGE_LH>() };
        let mut addrlen = mem::size_of_val(&storage) as c::socklen_t;
        let length = cmp::min(buf.len(), <wrlen_t>::MAX as usize) as wrlen_t;

        // On unix when a socket is shut down all further reads return 0, so we
        // do the same on windows to map a shut down socket to returning EOF.
        let result = unsafe {
            c::recvfrom(
                self.as_raw_socket(),
                buf.as_mut_ptr() as *mut _,
                length,
                flags,
                &mut storage as *mut _ as *mut _,
                &mut addrlen,
            )
        };

        match result {
            c::SOCKET_ERROR => {
                let error = unsafe { c::WSAGetLastError() };

                if error == c::WSAESHUTDOWN {
                    Ok((0, net::sockaddr_to_addr(&storage, addrlen as usize)?))
                } else {
                    Err(io::Error::from_raw_os_error(error))
                }
            }
            _ => Ok((result as usize, net::sockaddr_to_addr(&storage, addrlen as usize)?)),
        }
    }

    pub fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.recv_from_with_flags(buf, 0)
    }

    pub fn peek_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.recv_from_with_flags(buf, c::MSG_PEEK)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let length = cmp::min(bufs.len(), c::DWORD::MAX as usize) as c::DWORD;
        let mut nwritten = 0;
        let result = unsafe {
            c::WSASend(
                self.as_raw_socket(),
                bufs.as_ptr() as *const c::WSABUF as *mut _,
                length,
                &mut nwritten,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
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
                    return Err(io::Error::new_const(
                        io::ErrorKind::InvalidInput,
                        &"cannot set a 0 duration timeout",
                    ));
                }
                timeout
            }
            None => 0,
        };
        net::setsockopt(self, c::SOL_SOCKET, kind, timeout)
    }

    pub fn timeout(&self, kind: c_int) -> io::Result<Option<Duration>> {
        let raw: c::DWORD = net::getsockopt(self, c::SOL_SOCKET, kind)?;
        if raw == 0 {
            Ok(None)
        } else {
            let secs = raw / 1000;
            let nsec = (raw % 1000) * 1000000;
            Ok(Some(Duration::new(secs as u64, nsec as u32)))
        }
    }

    #[cfg(not(target_vendor = "uwp"))]
    fn set_no_inherit(&self) -> io::Result<()> {
        sys::cvt(unsafe {
            c::SetHandleInformation(self.as_raw_socket() as c::HANDLE, c::HANDLE_FLAG_INHERIT, 0)
        })
        .map(drop)
    }

    #[cfg(target_vendor = "uwp")]
    fn set_no_inherit(&self) -> io::Result<()> {
        Err(io::Error::new_const(io::ErrorKind::Unsupported, &"Unavailable on UWP"))
    }

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        let how = match how {
            Shutdown::Write => c::SD_SEND,
            Shutdown::Read => c::SD_RECEIVE,
            Shutdown::Both => c::SD_BOTH,
        };
        let result = unsafe { c::shutdown(self.as_raw_socket(), how) };
        cvt(result).map(drop)
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        let mut nonblocking = nonblocking as c_ulong;
        let result =
            unsafe { c::ioctlsocket(self.as_raw_socket(), c::FIONBIO as c_int, &mut nonblocking) };
        cvt(result).map(drop)
    }

    pub fn set_linger(&self, linger: Option<Duration>) -> io::Result<()> {
        let linger = c::linger {
            l_onoff: linger.is_some() as c_ushort,
            l_linger: linger.unwrap_or_default().as_secs() as c_ushort,
        };

        net::setsockopt(self, c::SOL_SOCKET, c::SO_LINGER, linger)
    }

    pub fn linger(&self) -> io::Result<Option<Duration>> {
        let val: c::linger = net::getsockopt(self, c::SOL_SOCKET, c::SO_LINGER)?;

        Ok((val.l_onoff != 0).then(|| Duration::from_secs(val.l_linger as u64)))
    }

    pub fn set_nodelay(&self, nodelay: bool) -> io::Result<()> {
        net::setsockopt(self, c::IPPROTO_TCP, c::TCP_NODELAY, nodelay as c::BYTE)
    }

    pub fn nodelay(&self) -> io::Result<bool> {
        let raw: c::BYTE = net::getsockopt(self, c::IPPROTO_TCP, c::TCP_NODELAY)?;
        Ok(raw != 0)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        let raw: c_int = net::getsockopt(self, c::SOL_SOCKET, c::SO_ERROR)?;
        if raw == 0 { Ok(None) } else { Ok(Some(io::Error::from_raw_os_error(raw as i32))) }
    }

    // This is used by sys_common code to abstract over Windows and Unix.
    pub fn as_raw(&self) -> RawSocket {
        self.as_inner().as_raw_socket()
    }
}

#[unstable(reason = "not public", issue = "none", feature = "fd_read")]
impl<'a> Read for &'a Socket {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        (**self).read(buf)
    }
}

impl AsInner<OwnedSocket> for Socket {
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
        Self(FromRawSocket::from_raw_socket(raw_socket))
    }
}
