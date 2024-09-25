use libc::{c_int, c_void, size_t};

use self::netc::{MSG_PEEK, sockaddr, socklen_t};
use super::abi;
use crate::ffi::CStr;
use crate::io::{self, BorrowedBuf, BorrowedCursor, ErrorKind, IoSlice, IoSliceMut};
use crate::net::{Shutdown, SocketAddr};
use crate::os::solid::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd};
use crate::sys_common::net::{getsockopt, setsockopt, sockaddr_to_addr};
use crate::sys_common::{FromInner, IntoInner};
use crate::time::Duration;
use crate::{cmp, mem, ptr, str};

pub mod netc {
    pub use super::super::abi::sockets::*;
}

pub type wrlen_t = size_t;

const READ_LIMIT: usize = libc::ssize_t::MAX as usize;

const fn max_iov() -> usize {
    // Judging by the source code, it's unlimited, but specify a lower
    // value just in case.
    1024
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

pub fn cvt<T: IsMinusOne>(t: T) -> io::Result<T> {
    if t.is_minus_one() { Err(last_error()) } else { Ok(t) }
}

/// A variant of `cvt` for `getaddrinfo` which return 0 for a success.
pub fn cvt_gai(err: c_int) -> io::Result<()> {
    if err == 0 {
        Ok(())
    } else {
        let msg: &dyn crate::fmt::Display = match err {
            netc::EAI_NONAME => &"name or service not known",
            netc::EAI_SERVICE => &"service not supported",
            netc::EAI_FAIL => &"non-recoverable failure in name resolution",
            netc::EAI_MEMORY => &"memory allocation failure",
            netc::EAI_FAMILY => &"family not supported",
            _ => &err,
        };
        Err(io::Error::new(
            io::ErrorKind::Uncategorized,
            &format!("failed to lookup address information: {msg}")[..],
        ))
    }
}

/// Just to provide the same interface as sys/pal/unix/net.rs
pub fn cvt_r<T, F>(mut f: F) -> io::Result<T>
where
    T: IsMinusOne,
    F: FnMut() -> T,
{
    cvt(f())
}

/// Returns the last error from the network subsystem.
fn last_error() -> io::Error {
    io::Error::from_raw_os_error(unsafe { netc::SOLID_NET_GetLastError() })
}

pub(super) fn error_name(er: abi::ER) -> Option<&'static str> {
    unsafe { CStr::from_ptr(netc::strerror(er)) }.to_str().ok()
}

#[inline]
pub fn is_interrupted(er: abi::ER) -> bool {
    er == netc::SOLID_NET_ERR_BASE - libc::EINTR
}

pub(super) fn decode_error_kind(er: abi::ER) -> ErrorKind {
    let errno = netc::SOLID_NET_ERR_BASE - er;
    match errno as libc::c_int {
        libc::ECONNREFUSED => ErrorKind::ConnectionRefused,
        libc::ECONNRESET => ErrorKind::ConnectionReset,
        libc::EPERM | libc::EACCES => ErrorKind::PermissionDenied,
        libc::EPIPE => ErrorKind::BrokenPipe,
        libc::ENOTCONN => ErrorKind::NotConnected,
        libc::ECONNABORTED => ErrorKind::ConnectionAborted,
        libc::EADDRNOTAVAIL => ErrorKind::AddrNotAvailable,
        libc::EADDRINUSE => ErrorKind::AddrInUse,
        libc::ENOENT => ErrorKind::NotFound,
        libc::EINTR => ErrorKind::Interrupted,
        libc::EINVAL => ErrorKind::InvalidInput,
        libc::ETIMEDOUT => ErrorKind::TimedOut,
        libc::EEXIST => ErrorKind::AlreadyExists,
        libc::ENOSYS => ErrorKind::Unsupported,
        libc::ENOMEM => ErrorKind::OutOfMemory,
        libc::EAGAIN => ErrorKind::WouldBlock,

        _ => ErrorKind::Uncategorized,
    }
}

pub fn init() {}

pub struct Socket(OwnedFd);

impl Socket {
    pub fn new(addr: &SocketAddr, ty: c_int) -> io::Result<Socket> {
        let fam = match *addr {
            SocketAddr::V4(..) => netc::AF_INET,
            SocketAddr::V6(..) => netc::AF_INET6,
        };
        Socket::new_raw(fam, ty)
    }

    pub fn new_raw(fam: c_int, ty: c_int) -> io::Result<Socket> {
        unsafe {
            let fd = cvt(netc::socket(fam, ty, 0))?;
            Ok(Self::from_raw_fd(fd))
        }
    }

    pub fn connect(&self, addr: &SocketAddr) -> io::Result<()> {
        let (addr, len) = addr.into_inner();
        cvt(unsafe { netc::connect(self.as_raw_fd(), addr.as_ptr(), len) })?;
        Ok(())
    }

    pub fn connect_timeout(&self, addr: &SocketAddr, timeout: Duration) -> io::Result<()> {
        self.set_nonblocking(true)?;
        let r = self.connect(addr);
        self.set_nonblocking(false)?;

        match r {
            Ok(_) => return Ok(()),
            // there's no ErrorKind for EINPROGRESS
            Err(ref e) if e.raw_os_error() == Some(netc::EINPROGRESS) => {}
            Err(e) => return Err(e),
        }

        if timeout.as_secs() == 0 && timeout.subsec_nanos() == 0 {
            return Err(io::Error::ZERO_TIMEOUT);
        }

        let mut timeout =
            netc::timeval { tv_sec: timeout.as_secs() as _, tv_usec: timeout.subsec_micros() as _ };
        if timeout.tv_sec == 0 && timeout.tv_usec == 0 {
            timeout.tv_usec = 1;
        }

        let fds = netc::fd_set { num_fds: 1, fds: [self.as_raw_fd()] };

        let mut writefds = fds;
        let mut errorfds = fds;

        let n = unsafe {
            cvt(netc::select(
                self.as_raw_fd() + 1,
                ptr::null_mut(),
                &mut writefds,
                &mut errorfds,
                &mut timeout,
            ))?
        };

        match n {
            0 => Err(io::const_io_error!(io::ErrorKind::TimedOut, "connection timed out")),
            _ => {
                let can_write = writefds.num_fds != 0;
                if !can_write {
                    if let Some(e) = self.take_error()? {
                        return Err(e);
                    }
                }
                Ok(())
            }
        }
    }

    pub fn accept(&self, storage: *mut sockaddr, len: *mut socklen_t) -> io::Result<Socket> {
        let fd = cvt_r(|| unsafe { netc::accept(self.as_raw_fd(), storage, len) })?;
        unsafe { Ok(Self::from_raw_fd(fd)) }
    }

    pub fn duplicate(&self) -> io::Result<Socket> {
        Ok(Self(self.0.try_clone()?))
    }

    fn recv_with_flags(&self, mut buf: BorrowedCursor<'_>, flags: c_int) -> io::Result<()> {
        let ret = cvt(unsafe {
            netc::recv(self.as_raw_fd(), buf.as_mut().as_mut_ptr().cast(), buf.capacity(), flags)
        })?;
        unsafe {
            buf.advance_unchecked(ret as usize);
        }
        Ok(())
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let mut buf = BorrowedBuf::from(buf);
        self.recv_with_flags(buf.unfilled(), 0)?;
        Ok(buf.len())
    }

    pub fn peek(&self, buf: &mut [u8]) -> io::Result<usize> {
        let mut buf = BorrowedBuf::from(buf);
        self.recv_with_flags(buf.unfilled(), MSG_PEEK)?;
        Ok(buf.len())
    }

    pub fn read_buf(&self, buf: BorrowedCursor<'_>) -> io::Result<()> {
        self.recv_with_flags(buf, 0)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            netc::readv(
                self.as_raw_fd(),
                bufs.as_ptr() as *const netc::iovec,
                cmp::min(bufs.len(), max_iov()) as c_int,
            )
        })?;
        Ok(ret as usize)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        true
    }

    fn recv_from_with_flags(
        &self,
        buf: &mut [u8],
        flags: c_int,
    ) -> io::Result<(usize, SocketAddr)> {
        let mut storage: netc::sockaddr_storage = unsafe { mem::zeroed() };
        let mut addrlen = mem::size_of_val(&storage) as netc::socklen_t;

        let n = cvt(unsafe {
            netc::recvfrom(
                self.as_raw_fd(),
                buf.as_mut_ptr() as *mut c_void,
                buf.len(),
                flags,
                &mut storage as *mut _ as *mut _,
                &mut addrlen,
            )
        })?;
        Ok((n as usize, sockaddr_to_addr(&storage, addrlen as usize)?))
    }

    pub fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.recv_from_with_flags(buf, 0)
    }

    pub fn peek_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.recv_from_with_flags(buf, MSG_PEEK)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            netc::write(
                self.as_raw_fd(),
                buf.as_ptr() as *const c_void,
                cmp::min(buf.len(), READ_LIMIT),
            )
        })?;
        Ok(ret as usize)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            netc::writev(
                self.as_raw_fd(),
                bufs.as_ptr() as *const netc::iovec,
                cmp::min(bufs.len(), max_iov()) as c_int,
            )
        })?;
        Ok(ret as usize)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        true
    }

    pub fn set_timeout(&self, dur: Option<Duration>, kind: c_int) -> io::Result<()> {
        let timeout = match dur {
            Some(dur) => {
                if dur.as_secs() == 0 && dur.subsec_nanos() == 0 {
                    return Err(io::Error::ZERO_TIMEOUT);
                }

                let secs = if dur.as_secs() > netc::c_long::MAX as u64 {
                    netc::c_long::MAX
                } else {
                    dur.as_secs() as netc::c_long
                };
                let mut timeout = netc::timeval { tv_sec: secs, tv_usec: dur.subsec_micros() as _ };
                if timeout.tv_sec == 0 && timeout.tv_usec == 0 {
                    timeout.tv_usec = 1;
                }
                timeout
            }
            None => netc::timeval { tv_sec: 0, tv_usec: 0 },
        };
        setsockopt(self, netc::SOL_SOCKET, kind, timeout)
    }

    pub fn timeout(&self, kind: c_int) -> io::Result<Option<Duration>> {
        let raw: netc::timeval = getsockopt(self, netc::SOL_SOCKET, kind)?;
        if raw.tv_sec == 0 && raw.tv_usec == 0 {
            Ok(None)
        } else {
            let sec = raw.tv_sec as u64;
            let nsec = (raw.tv_usec as u32) * 1000;
            Ok(Some(Duration::new(sec, nsec)))
        }
    }

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        let how = match how {
            Shutdown::Write => netc::SHUT_WR,
            Shutdown::Read => netc::SHUT_RD,
            Shutdown::Both => netc::SHUT_RDWR,
        };
        cvt(unsafe { netc::shutdown(self.as_raw_fd(), how) })?;
        Ok(())
    }

    pub fn set_linger(&self, linger: Option<Duration>) -> io::Result<()> {
        let linger = netc::linger {
            l_onoff: linger.is_some() as netc::c_int,
            l_linger: linger.unwrap_or_default().as_secs() as netc::c_int,
        };

        setsockopt(self, netc::SOL_SOCKET, netc::SO_LINGER, linger)
    }

    pub fn linger(&self) -> io::Result<Option<Duration>> {
        let val: netc::linger = getsockopt(self, netc::SOL_SOCKET, netc::SO_LINGER)?;

        Ok((val.l_onoff != 0).then(|| Duration::from_secs(val.l_linger as u64)))
    }

    pub fn set_nodelay(&self, nodelay: bool) -> io::Result<()> {
        setsockopt(self, netc::IPPROTO_TCP, netc::TCP_NODELAY, nodelay as c_int)
    }

    pub fn nodelay(&self) -> io::Result<bool> {
        let raw: c_int = getsockopt(self, netc::IPPROTO_TCP, netc::TCP_NODELAY)?;
        Ok(raw != 0)
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        let mut nonblocking = nonblocking as c_int;
        cvt(unsafe {
            netc::ioctl(self.as_raw_fd(), netc::FIONBIO, (&mut nonblocking) as *mut c_int as _)
        })
        .map(drop)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        let raw: c_int = getsockopt(self, netc::SOL_SOCKET, netc::SO_ERROR)?;
        if raw == 0 { Ok(None) } else { Ok(Some(io::Error::from_raw_os_error(raw as i32))) }
    }

    // This method is used by sys_common code to abstract over targets.
    pub fn as_raw(&self) -> c_int {
        self.as_raw_fd()
    }
}

impl FromInner<OwnedFd> for Socket {
    #[inline]
    fn from_inner(sock: OwnedFd) -> Socket {
        Socket(sock)
    }
}

impl IntoInner<OwnedFd> for Socket {
    #[inline]
    fn into_inner(self) -> OwnedFd {
        self.0
    }
}

impl AsFd for Socket {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

impl AsRawFd for Socket {
    #[inline]
    fn as_raw_fd(&self) -> c_int {
        self.0.as_raw_fd()
    }
}

impl FromRawFd for Socket {
    #[inline]
    unsafe fn from_raw_fd(fd: c_int) -> Socket {
        unsafe { Self(FromRawFd::from_raw_fd(fd)) }
    }
}

impl IntoRawFd for Socket {
    #[inline]
    fn into_raw_fd(self) -> c_int {
        self.0.into_raw_fd()
    }
}
