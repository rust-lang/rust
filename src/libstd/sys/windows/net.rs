#![unstable(issue = "0", feature = "windows_net")]

use crate::cmp;
use crate::io::{self, Read, IoSlice, IoSliceMut};
use crate::mem;
use crate::net::{SocketAddr, Shutdown};
use crate::ptr;
use crate::sync::Once;
use crate::sys::c;
use crate::sys;
use crate::sys_common::{self, AsInner, FromInner, IntoInner};
use crate::sys_common::net;
use crate::time::Duration;

use libc::{c_int, c_void, c_ulong, c_long};

pub type wrlen_t = i32;

pub mod netc {
    pub use crate::sys::c::*;
    pub use crate::sys::c::SOCKADDR as sockaddr;
    pub use crate::sys::c::SOCKADDR_STORAGE_LH as sockaddr_storage;
    pub use crate::sys::c::ADDRINFOA as addrinfo;
    pub use crate::sys::c::ADDRESS_FAMILY as sa_family_t;
}

pub struct Socket(c::SOCKET);

/// Checks whether the Windows socket interface has been started already, and
/// if not, starts it.
pub fn init() {
    static START: Once = Once::new();

    START.call_once(|| unsafe {
        let mut data: c::WSADATA = mem::zeroed();
        let ret = c::WSAStartup(0x202, // version 2.2
                                &mut data);
        assert_eq!(ret, 0);

        let _ = sys_common::at_exit(|| { c::WSACleanup(); });
    });
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
    if t.is_minus_one() {
        Err(last_error())
    } else {
        Ok(t)
    }
}

/// A variant of `cvt` for `getaddrinfo` which return 0 for a success.
pub fn cvt_gai(err: c_int) -> io::Result<()> {
    if err == 0 {
        Ok(())
    } else {
        Err(last_error())
    }
}

/// Just to provide the same interface as sys/unix/net.rs
pub fn cvt_r<T, F>(mut f: F) -> io::Result<T>
    where T: IsMinusOne,
          F: FnMut() -> T
{
    cvt(f())
}

impl Socket {
    pub fn new(addr: &SocketAddr, ty: c_int) -> io::Result<Socket> {
        let fam = match *addr {
            SocketAddr::V4(..) => c::AF_INET,
            SocketAddr::V6(..) => c::AF_INET6,
        };
        let socket = unsafe {
            match c::WSASocketW(fam, ty, 0, ptr::null_mut(), 0,
                                c::WSA_FLAG_OVERLAPPED) {
                c::INVALID_SOCKET => Err(last_error()),
                n => Ok(Socket(n)),
            }
        }?;
        socket.set_no_inherit()?;
        Ok(socket)
    }

    pub fn connect_timeout(&self, addr: &SocketAddr, timeout: Duration) -> io::Result<()> {
        self.set_nonblocking(true)?;
        let r = unsafe {
            let (addrp, len) = addr.into_inner();
            cvt(c::connect(self.0, addrp, len))
        };
        self.set_nonblocking(false)?;

        match r {
            Ok(_) => return Ok(()),
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {}
            Err(e) => return Err(e),
        }

        if timeout.as_secs() == 0 && timeout.subsec_nanos() == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput,
                                      "cannot set a 0 duration timeout"));
        }

        let mut timeout = c::timeval {
            tv_sec: timeout.as_secs() as c_long,
            tv_usec: (timeout.subsec_nanos() / 1000) as c_long,
        };
        if timeout.tv_sec == 0 && timeout.tv_usec == 0 {
            timeout.tv_usec = 1;
        }

        let fds = unsafe {
            let mut fds = mem::zeroed::<c::fd_set>();
            fds.fd_count = 1;
            fds.fd_array[0] = self.0;
            fds
        };

        let mut writefds = fds;
        let mut errorfds = fds;

        let n = unsafe {
            cvt(c::select(1, ptr::null_mut(), &mut writefds, &mut errorfds, &timeout))?
        };

        match n {
            0 => Err(io::Error::new(io::ErrorKind::TimedOut, "connection timed out")),
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

    pub fn accept(&self, storage: *mut c::SOCKADDR,
                  len: *mut c_int) -> io::Result<Socket> {
        let socket = unsafe {
            match c::accept(self.0, storage, len) {
                c::INVALID_SOCKET => Err(last_error()),
                n => Ok(Socket(n)),
            }
        }?;
        socket.set_no_inherit()?;
        Ok(socket)
    }

    pub fn duplicate(&self) -> io::Result<Socket> {
        let socket = unsafe {
            let mut info: c::WSAPROTOCOL_INFO = mem::zeroed();
            cvt(c::WSADuplicateSocketW(self.0,
                                            c::GetCurrentProcessId(),
                                            &mut info))?;
            match c::WSASocketW(info.iAddressFamily,
                                info.iSocketType,
                                info.iProtocol,
                                &mut info, 0,
                                c::WSA_FLAG_OVERLAPPED) {
                c::INVALID_SOCKET => Err(last_error()),
                n => Ok(Socket(n)),
            }
        }?;
        socket.set_no_inherit()?;
        Ok(socket)
    }

    fn recv_with_flags(&self, buf: &mut [u8], flags: c_int) -> io::Result<usize> {
        // On unix when a socket is shut down all further reads return 0, so we
        // do the same on windows to map a shut down socket to returning EOF.
        let len = cmp::min(buf.len(), i32::max_value() as usize) as i32;
        unsafe {
            match c::recv(self.0, buf.as_mut_ptr() as *mut c_void, len, flags) {
                -1 if c::WSAGetLastError() == c::WSAESHUTDOWN => Ok(0),
                -1 => Err(last_error()),
                n => Ok(n as usize)
            }
        }
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.recv_with_flags(buf, 0)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        // On unix when a socket is shut down all further reads return 0, so we
        // do the same on windows to map a shut down socket to returning EOF.
        let len = cmp::min(bufs.len(), c::DWORD::max_value() as usize) as c::DWORD;
        let mut nread = 0;
        let mut flags = 0;
        unsafe {
            let ret = c::WSARecv(
                self.0,
                bufs.as_mut_ptr() as *mut c::WSABUF,
                len,
                &mut nread,
                &mut flags,
                ptr::null_mut(),
                ptr::null_mut(),
            );
            match ret {
                0 => Ok(nread as usize),
                _ if c::WSAGetLastError() == c::WSAESHUTDOWN => Ok(0),
                _ => Err(last_error()),
            }
        }
    }

    pub fn peek(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.recv_with_flags(buf, c::MSG_PEEK)
    }

    fn recv_from_with_flags(&self, buf: &mut [u8], flags: c_int)
                            -> io::Result<(usize, SocketAddr)> {
        let mut storage: c::SOCKADDR_STORAGE_LH = unsafe { mem::zeroed() };
        let mut addrlen = mem::size_of_val(&storage) as c::socklen_t;
        let len = cmp::min(buf.len(), <wrlen_t>::max_value() as usize) as wrlen_t;

        // On unix when a socket is shut down all further reads return 0, so we
        // do the same on windows to map a shut down socket to returning EOF.
        unsafe {
            match c::recvfrom(self.0,
                              buf.as_mut_ptr() as *mut c_void,
                              len,
                              flags,
                              &mut storage as *mut _ as *mut _,
                              &mut addrlen) {
                -1 if c::WSAGetLastError() == c::WSAESHUTDOWN => {
                    Ok((0, net::sockaddr_to_addr(&storage, addrlen as usize)?))
                },
                -1 => Err(last_error()),
                n => Ok((n as usize, net::sockaddr_to_addr(&storage, addrlen as usize)?)),
            }
        }
    }

    pub fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.recv_from_with_flags(buf, 0)
    }

    pub fn peek_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.recv_from_with_flags(buf, c::MSG_PEEK)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let len = cmp::min(bufs.len(), c::DWORD::max_value() as usize) as c::DWORD;
        let mut nwritten = 0;
        unsafe {
            cvt(c::WSASend(
                self.0,
                bufs.as_ptr() as *const c::WSABUF as *mut c::WSABUF,
                len,
                &mut nwritten,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            ))?;
        }
        Ok(nwritten as usize)
    }

    pub fn set_timeout(&self, dur: Option<Duration>,
                       kind: c_int) -> io::Result<()> {
        let timeout = match dur {
            Some(dur) => {
                let timeout = sys::dur2timeout(dur);
                if timeout == 0 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput,
                                              "cannot set a 0 duration timeout"));
                }
                timeout
            }
            None => 0
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

    fn set_no_inherit(&self) -> io::Result<()> {
        sys::cvt(unsafe {
            c::SetHandleInformation(self.0 as c::HANDLE,
                                    c::HANDLE_FLAG_INHERIT, 0)
        }).map(|_| ())
    }

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        let how = match how {
            Shutdown::Write => c::SD_SEND,
            Shutdown::Read => c::SD_RECEIVE,
            Shutdown::Both => c::SD_BOTH,
        };
        cvt(unsafe { c::shutdown(self.0, how) })?;
        Ok(())
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        let mut nonblocking = nonblocking as c_ulong;
        let r = unsafe { c::ioctlsocket(self.0, c::FIONBIO as c_int, &mut nonblocking) };
        if r == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
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
        if raw == 0 {
            Ok(None)
        } else {
            Ok(Some(io::Error::from_raw_os_error(raw as i32)))
        }
    }
}

#[unstable(reason = "not public", issue = "0", feature = "fd_read")]
impl<'a> Read for &'a Socket {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        (**self).read(buf)
    }
}

impl Drop for Socket {
    fn drop(&mut self) {
        let _ = unsafe { c::closesocket(self.0) };
    }
}

impl AsInner<c::SOCKET> for Socket {
    fn as_inner(&self) -> &c::SOCKET { &self.0 }
}

impl FromInner<c::SOCKET> for Socket {
    fn from_inner(sock: c::SOCKET) -> Socket { Socket(sock) }
}

impl IntoInner<c::SOCKET> for Socket {
    fn into_inner(self) -> c::SOCKET {
        let ret = self.0;
        mem::forget(self);
        ret
    }
}
