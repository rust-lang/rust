use super::abi;
use crate::{
    cmp,
    ffi::CStr,
    io::{self, ErrorKind, IoSlice, IoSliceMut},
    mem,
    net::{Shutdown, SocketAddr},
    ptr, str,
    sys_common::net::{getsockopt, setsockopt, sockaddr_to_addr},
    sys_common::{AsInner, FromInner, IntoInner},
    time::Duration,
};

use self::netc::{sockaddr, socklen_t, MSG_PEEK};
use libc::{c_int, c_void, size_t};

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

/// A file descriptor.
#[rustc_layout_scalar_valid_range_start(0)]
// libstd/os/raw/mod.rs assures me that every libstd-supported platform has a
// 32-bit c_int. Below is -2, in two's complement, but that only works out
// because c_int is 32 bits.
#[rustc_layout_scalar_valid_range_end(0xFF_FF_FF_FE)]
struct FileDesc {
    fd: c_int,
}

impl FileDesc {
    #[inline]
    fn new(fd: c_int) -> FileDesc {
        assert_ne!(fd, -1i32);
        // Safety: we just asserted that the value is in the valid range and
        // isn't `-1` (the only value bigger than `0xFF_FF_FF_FE` unsigned)
        unsafe { FileDesc { fd } }
    }

    #[inline]
    fn raw(&self) -> c_int {
        self.fd
    }

    /// Extracts the actual file descriptor without closing it.
    #[inline]
    fn into_raw(self) -> c_int {
        let fd = self.fd;
        mem::forget(self);
        fd
    }

    fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            netc::read(self.fd, buf.as_mut_ptr() as *mut c_void, cmp::min(buf.len(), READ_LIMIT))
        })?;
        Ok(ret as usize)
    }

    fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            netc::readv(
                self.fd,
                bufs.as_ptr() as *const netc::iovec,
                cmp::min(bufs.len(), max_iov()) as c_int,
            )
        })?;
        Ok(ret as usize)
    }

    #[inline]
    fn is_read_vectored(&self) -> bool {
        true
    }

    fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            netc::write(self.fd, buf.as_ptr() as *const c_void, cmp::min(buf.len(), READ_LIMIT))
        })?;
        Ok(ret as usize)
    }

    fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            netc::writev(
                self.fd,
                bufs.as_ptr() as *const netc::iovec,
                cmp::min(bufs.len(), max_iov()) as c_int,
            )
        })?;
        Ok(ret as usize)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        true
    }

    fn duplicate(&self) -> io::Result<FileDesc> {
        super::unsupported()
    }
}

impl AsInner<c_int> for FileDesc {
    fn as_inner(&self) -> &c_int {
        &self.fd
    }
}

impl Drop for FileDesc {
    fn drop(&mut self) {
        unsafe { netc::close(self.fd) };
    }
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
            &format!("failed to lookup address information: {}", msg)[..],
        ))
    }
}

/// Just to provide the same interface as sys/unix/net.rs
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

pub struct Socket(FileDesc);

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
            let fd = FileDesc::new(fd);
            let socket = Socket(fd);

            Ok(socket)
        }
    }

    pub fn connect_timeout(&self, addr: &SocketAddr, timeout: Duration) -> io::Result<()> {
        self.set_nonblocking(true)?;
        let r = unsafe {
            let (addrp, len) = addr.into_inner();
            cvt(netc::connect(self.0.raw(), addrp, len))
        };
        self.set_nonblocking(false)?;

        match r {
            Ok(_) => return Ok(()),
            // there's no ErrorKind for EINPROGRESS
            Err(ref e) if e.raw_os_error() == Some(netc::EINPROGRESS) => {}
            Err(e) => return Err(e),
        }

        if timeout.as_secs() == 0 && timeout.subsec_nanos() == 0 {
            return Err(io::Error::new_const(
                io::ErrorKind::InvalidInput,
                &"cannot set a 0 duration timeout",
            ));
        }

        let mut timeout =
            netc::timeval { tv_sec: timeout.as_secs() as _, tv_usec: timeout.subsec_micros() as _ };
        if timeout.tv_sec == 0 && timeout.tv_usec == 0 {
            timeout.tv_usec = 1;
        }

        let fds = netc::fd_set { num_fds: 1, fds: [self.0.raw()] };

        let mut writefds = fds;
        let mut errorfds = fds;

        let n = unsafe {
            cvt(netc::select(
                self.0.raw() + 1,
                ptr::null_mut(),
                &mut writefds,
                &mut errorfds,
                &mut timeout,
            ))?
        };

        match n {
            0 => Err(io::Error::new_const(io::ErrorKind::TimedOut, &"connection timed out")),
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
        let fd = cvt_r(|| unsafe { netc::accept(self.0.raw(), storage, len) })?;
        let fd = FileDesc::new(fd);
        Ok(Socket(fd))
    }

    pub fn duplicate(&self) -> io::Result<Socket> {
        self.0.duplicate().map(Socket)
    }

    fn recv_with_flags(&self, buf: &mut [u8], flags: c_int) -> io::Result<usize> {
        let ret = cvt(unsafe {
            netc::recv(self.0.raw(), buf.as_mut_ptr() as *mut c_void, buf.len(), flags)
        })?;
        Ok(ret as usize)
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.recv_with_flags(buf, 0)
    }

    pub fn peek(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.recv_with_flags(buf, MSG_PEEK)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.0.read_vectored(bufs)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        self.0.is_read_vectored()
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
                self.0.raw(),
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
        self.0.write(buf)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.0.write_vectored(bufs)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        self.0.is_write_vectored()
    }

    pub fn set_timeout(&self, dur: Option<Duration>, kind: c_int) -> io::Result<()> {
        let timeout = match dur {
            Some(dur) => {
                if dur.as_secs() == 0 && dur.subsec_nanos() == 0 {
                    return Err(io::Error::new_const(
                        io::ErrorKind::InvalidInput,
                        &"cannot set a 0 duration timeout",
                    ));
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
        cvt(unsafe { netc::shutdown(self.0.raw(), how) })?;
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
            netc::ioctl(*self.as_inner(), netc::FIONBIO, (&mut nonblocking) as *mut c_int as _)
        })
        .map(drop)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        let raw: c_int = getsockopt(self, netc::SOL_SOCKET, netc::SO_ERROR)?;
        if raw == 0 { Ok(None) } else { Ok(Some(io::Error::from_raw_os_error(raw as i32))) }
    }

    // This method is used by sys_common code to abstract over targets.
    pub fn as_raw(&self) -> c_int {
        *self.as_inner()
    }
}

impl AsInner<c_int> for Socket {
    fn as_inner(&self) -> &c_int {
        self.0.as_inner()
    }
}

impl FromInner<c_int> for Socket {
    fn from_inner(fd: c_int) -> Socket {
        Socket(FileDesc::new(fd))
    }
}

impl IntoInner<c_int> for Socket {
    fn into_inner(self) -> c_int {
        self.0.into_raw()
    }
}
