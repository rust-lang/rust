#![deny(unsafe_op_in_unsafe_fn)]

use libc::{c_int, c_void, size_t};

use crate::ffi::CStr;
use crate::io::{self, BorrowedBuf, BorrowedCursor, IoSlice, IoSliceMut};
use crate::net::{Shutdown, SocketAddr};
use crate::os::wasi::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use crate::sys::unsupported;
use crate::sys_common::net::{getsockopt, setsockopt, sockaddr_to_addr};
use crate::sys_common::{AsInner, FromInner, IntoInner};
use crate::time::{Duration, Instant};
use crate::{cmp, mem, str};

pub extern crate libc as netc;

#[allow(non_camel_case_types)]
pub type wrlen_t = size_t;

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

pub fn cvt<T: IsMinusOne>(t: T) -> crate::io::Result<T> {
    if t.is_minus_one() { Err(crate::io::Error::last_os_error()) } else { Ok(t) }
}

pub fn cvt_r<T, F>(mut f: F) -> crate::io::Result<T>
where
    T: IsMinusOne,
    F: FnMut() -> T,
{
    loop {
        match cvt(f()) {
            Err(ref e) if e.is_interrupted() => {}
            other => return other,
        }
    }
}

pub fn cvt_gai(err: c_int) -> io::Result<()> {
    if err == 0 {
        return Ok(());
    }

    if err == netc::EAI_SYSTEM {
        return Err(io::Error::last_os_error());
    }

    let detail = unsafe {
        str::from_utf8(CStr::from_ptr(netc::gai_strerror(err)).to_bytes()).unwrap().to_owned()
    };

    Err(io::Error::new(
        io::ErrorKind::Uncategorized,
        &format!("failed to lookup address information: {detail}")[..],
    ))
}

pub fn init() {}

pub struct WasiSocket(OwnedFd);

pub struct Socket(WasiSocket);

impl Socket {
    pub fn new(addr: &SocketAddr, ty: c_int) -> io::Result<Socket> {
        let fam = match *addr {
            SocketAddr::V4(..) => netc::AF_INET,
            SocketAddr::V6(..) => netc::AF_INET6,
        };
        Socket::new_raw(fam, ty)
    }

    pub fn new_raw(fam: c_int, ty: c_int) -> io::Result<Socket> {
        let fd = cvt(unsafe { netc::socket(fam, ty, 0) })?;
        Ok(unsafe { Self::from_raw_fd(fd) })
    }

    pub fn connect(&self, addr: &SocketAddr) -> io::Result<()> {
        let (addr, len) = addr.into_inner();
        cvt_r(|| unsafe { netc::connect(self.as_raw_fd(), addr.as_ptr(), len) })?;
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

        let mut pollfd = netc::pollfd { fd: self.as_raw_fd(), events: netc::POLLOUT, revents: 0 };

        if timeout.as_secs() == 0 && timeout.subsec_nanos() == 0 {
            return Err(io::Error::ZERO_TIMEOUT);
        }

        let start = Instant::now();

        loop {
            let elapsed = start.elapsed();
            if elapsed >= timeout {
                return Err(io::const_io_error!(io::ErrorKind::TimedOut, "connection timed out"));
            }

            let timeout = timeout - elapsed;
            let mut timeout = timeout
                .as_secs()
                .saturating_mul(1_000)
                .saturating_add(timeout.subsec_nanos() as u64 / 1_000_000);
            if timeout == 0 {
                timeout = 1;
            }

            let timeout = cmp::min(timeout, c_int::MAX as u64) as c_int;

            match unsafe { netc::poll(&mut pollfd, 1, timeout) } {
                -1 => {
                    let err = io::Error::last_os_error();
                    if !err.is_interrupted() {
                        return Err(err);
                    }
                }
                0 => {}
                _ => {
                    // WASI poll does not return  POLLHUP or POLLERR in revents. Check if the
                    // connnection actually succeeded and return ok only when the socket is
                    // ready and no errors were found.
                    if let Some(e) = self.take_error()? {
                        return Err(e);
                    }

                    return Ok(());
                }
            }
        }
    }

    pub fn accept(
        &self,
        storage: *mut netc::sockaddr,
        len: *mut netc::socklen_t,
    ) -> io::Result<Socket> {
        let fd = cvt_r(|| unsafe { netc::accept(self.as_raw_fd(), storage, len) })?;
        Ok(unsafe { Self::from_raw_fd(fd) })
    }

    pub fn duplicate(&self) -> io::Result<Socket> {
        unsupported()
    }

    fn recv_with_flags(&self, mut buf: BorrowedCursor<'_>, flags: c_int) -> io::Result<()> {
        let ret = cvt(unsafe {
            netc::recv(
                self.as_raw_fd(),
                buf.as_mut().as_mut_ptr() as *mut c_void,
                buf.capacity(),
                flags,
            )
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
        self.recv_with_flags(buf.unfilled(), netc::MSG_PEEK)?;
        Ok(buf.len())
    }

    pub fn read_buf(&self, buf: BorrowedCursor<'_>) -> io::Result<()> {
        self.recv_with_flags(buf, 0)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        io::default_read_vectored(|b| self.read(b), bufs)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        false
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
                core::ptr::addr_of_mut!(storage) as *mut _,
                &mut addrlen,
            )
        })?;
        Ok((n as usize, sockaddr_to_addr(&storage, addrlen as usize)?))
    }

    pub fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.recv_from_with_flags(buf, 0)
    }

    pub fn peek_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.recv_from_with_flags(buf, netc::MSG_PEEK)
    }

    fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let len = cmp::min(buf.len(), <wrlen_t>::MAX as usize) as wrlen_t;
        let ret = cvt(unsafe {
            netc::send(self.as_raw(), buf.as_ptr() as *const c_void, len, netc::MSG_NOSIGNAL)
        })?;
        Ok(ret as usize)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        io::default_write_vectored(|b| self.write(b), bufs)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        false
    }

    pub fn set_timeout(&self, dur: Option<Duration>, kind: c_int) -> io::Result<()> {
        let timeout = match dur {
            Some(dur) => {
                if dur.as_secs() == 0 && dur.subsec_nanos() == 0 {
                    return Err(io::Error::ZERO_TIMEOUT);
                }

                let secs = dur.as_secs().try_into().unwrap_or(netc::time_t::MAX);
                let mut timeout = netc::timeval {
                    tv_sec: secs,
                    tv_usec: dur.subsec_micros() as netc::suseconds_t,
                };
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

    pub fn set_linger(&self, _linger: Option<Duration>) -> io::Result<()> {
        unsupported()
    }

    pub fn linger(&self) -> io::Result<Option<Duration>> {
        unsupported()
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
        cvt(unsafe { netc::ioctl(self.as_raw_fd(), netc::FIONBIO, &mut nonblocking) }).map(drop)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        let raw: c_int = getsockopt(self, netc::SOL_SOCKET, netc::SO_ERROR)?;
        if raw == 0 { Ok(None) } else { Ok(Some(io::Error::from_raw_os_error(raw as i32))) }
    }

    // This is used by sys_common code to abstract over Windows and Unix.
    pub fn as_raw(&self) -> RawFd {
        self.as_raw_fd()
    }
}

impl AsInner<OwnedFd> for WasiSocket {
    #[inline]
    fn as_inner(&self) -> &OwnedFd {
        &self.0
    }
}

impl IntoInner<OwnedFd> for WasiSocket {
    fn into_inner(self) -> OwnedFd {
        self.0
    }
}

impl FromInner<OwnedFd> for WasiSocket {
    fn from_inner(owned_fd: OwnedFd) -> Self {
        Self(owned_fd)
    }
}

impl AsFd for WasiSocket {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

impl AsRawFd for WasiSocket {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.0.as_raw_fd()
    }
}

impl IntoRawFd for WasiSocket {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_raw_fd()
    }
}

impl FromRawFd for WasiSocket {
    unsafe fn from_raw_fd(raw_fd: RawFd) -> Self {
        unsafe { Self(FromRawFd::from_raw_fd(raw_fd)) }
    }
}

impl AsInner<WasiSocket> for Socket {
    #[inline]
    fn as_inner(&self) -> &WasiSocket {
        &self.0
    }
}

impl IntoInner<WasiSocket> for Socket {
    fn into_inner(self) -> WasiSocket {
        self.0
    }
}

impl FromInner<WasiSocket> for Socket {
    fn from_inner(sock: WasiSocket) -> Socket {
        Socket(sock)
    }
}

impl AsFd for Socket {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

impl AsRawFd for Socket {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.0.as_raw_fd()
    }
}

impl IntoRawFd for Socket {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_raw_fd()
    }
}

impl FromRawFd for Socket {
    unsafe fn from_raw_fd(raw_fd: RawFd) -> Self {
        unsafe { Self(FromRawFd::from_raw_fd(raw_fd)) }
    }
}
