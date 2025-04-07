#![allow(dead_code)]

use core::ffi::c_int;

pub(super) use hermit_abi as netc;

use super::{getsockopt, setsockopt, socket_addr_from_c, socket_addr_to_c};
use crate::io::{self, BorrowedBuf, BorrowedCursor, IoSlice, IoSliceMut};
use crate::net::{Shutdown, SocketAddr};
use crate::os::hermit::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, RawFd};
use crate::sys::fd::FileDesc;
use crate::sys::time::Instant;
pub use crate::sys::{cvt, cvt_r};
use crate::sys_common::{AsInner, FromInner, IntoInner};
use crate::time::Duration;
use crate::{cmp, mem};

#[expect(non_camel_case_types)]
pub type wrlen_t = usize;

pub fn cvt_gai(err: i32) -> io::Result<()> {
    if err == 0 {
        return Ok(());
    }

    let detail = "";

    Err(io::Error::new(
        io::ErrorKind::Uncategorized,
        &format!("failed to lookup address information: {detail}")[..],
    ))
}

pub fn init() {}

#[derive(Debug)]
pub struct Socket(FileDesc);

impl Socket {
    pub fn new(addr: &SocketAddr, ty: i32) -> io::Result<Socket> {
        let fam = match *addr {
            SocketAddr::V4(..) => netc::AF_INET,
            SocketAddr::V6(..) => netc::AF_INET6,
        };
        Socket::new_raw(fam, ty)
    }

    pub fn new_raw(fam: i32, ty: i32) -> io::Result<Socket> {
        let fd = cvt(unsafe { netc::socket(fam, ty, 0) })?;
        Ok(Socket(unsafe { FileDesc::from_raw_fd(fd) }))
    }

    pub fn new_pair(_fam: i32, _ty: i32) -> io::Result<(Socket, Socket)> {
        unimplemented!()
    }

    pub fn connect(&self, addr: &SocketAddr) -> io::Result<()> {
        let (addr, len) = socket_addr_to_c(addr);
        cvt_r(|| unsafe { netc::connect(self.as_raw_fd(), addr.as_ptr(), len) })?;
        Ok(())
    }

    pub fn connect_timeout(&self, addr: &SocketAddr, timeout: Duration) -> io::Result<()> {
        self.set_nonblocking(true)?;
        let r = unsafe {
            let (addr, len) = socket_addr_to_c(addr);
            cvt(netc::connect(self.as_raw_fd(), addr.as_ptr(), len))
        };
        self.set_nonblocking(false)?;

        match r {
            Ok(_) => return Ok(()),
            // there's no ErrorKind for EINPROGRESS :(
            Err(ref e) if e.raw_os_error() == Some(netc::errno::EINPROGRESS) => {}
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
                return Err(io::const_error!(io::ErrorKind::TimedOut, "connection timed out"));
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
                    // linux returns POLLOUT|POLLERR|POLLHUP for refused connections (!), so look
                    // for POLLHUP rather than read readiness
                    if pollfd.revents & netc::POLLHUP != 0 {
                        let e = self.take_error()?.unwrap_or_else(|| {
                            io::const_error!(
                                io::ErrorKind::Uncategorized,
                                "no error set after POLLHUP",
                            )
                        });
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
        let fd = cvt(unsafe { netc::accept(self.0.as_raw_fd(), storage, len) })?;
        Ok(Socket(unsafe { FileDesc::from_raw_fd(fd) }))
    }

    pub fn duplicate(&self) -> io::Result<Socket> {
        let fd = cvt(unsafe { netc::dup(self.0.as_raw_fd()) })?;
        Ok(Socket(unsafe { FileDesc::from_raw_fd(fd) }))
    }

    fn recv_with_flags(&self, mut buf: BorrowedCursor<'_>, flags: i32) -> io::Result<()> {
        let ret = cvt(unsafe {
            netc::recv(
                self.0.as_raw_fd(),
                buf.as_mut().as_mut_ptr() as *mut u8,
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
        self.0.read_vectored(bufs)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        self.0.is_read_vectored()
    }

    fn recv_from_with_flags(&self, buf: &mut [u8], flags: i32) -> io::Result<(usize, SocketAddr)> {
        let mut storage: netc::sockaddr_storage = unsafe { mem::zeroed() };
        let mut addrlen = size_of_val(&storage) as netc::socklen_t;

        let n = cvt(unsafe {
            netc::recvfrom(
                self.as_raw_fd(),
                buf.as_mut_ptr(),
                buf.len(),
                flags,
                (&raw mut storage) as *mut _,
                &mut addrlen,
            )
        })?;
        Ok((n as usize, unsafe { socket_addr_from_c(&storage, addrlen as usize)? }))
    }

    pub fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.recv_from_with_flags(buf, 0)
    }

    pub fn peek_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.recv_from_with_flags(buf, netc::MSG_PEEK)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.0.write_vectored(bufs)
    }

    pub fn is_write_vectored(&self) -> bool {
        self.0.is_write_vectored()
    }

    pub fn set_timeout(&self, dur: Option<Duration>, kind: i32) -> io::Result<()> {
        let timeout = match dur {
            Some(dur) => {
                if dur.as_secs() == 0 && dur.subsec_nanos() == 0 {
                    return Err(io::Error::ZERO_TIMEOUT);
                }

                let secs = if dur.as_secs() > netc::time_t::MAX as u64 {
                    netc::time_t::MAX
                } else {
                    dur.as_secs() as netc::time_t
                };
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

    pub fn timeout(&self, kind: i32) -> io::Result<Option<Duration>> {
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
            l_onoff: linger.is_some() as i32,
            l_linger: linger.unwrap_or_default().as_secs() as libc::c_int,
        };

        setsockopt(self, netc::SOL_SOCKET, netc::SO_LINGER, linger)
    }

    pub fn linger(&self) -> io::Result<Option<Duration>> {
        let val: netc::linger = getsockopt(self, netc::SOL_SOCKET, netc::SO_LINGER)?;

        Ok((val.l_onoff != 0).then(|| Duration::from_secs(val.l_linger as u64)))
    }

    pub fn set_nodelay(&self, nodelay: bool) -> io::Result<()> {
        let value: i32 = if nodelay { 1 } else { 0 };
        setsockopt(self, netc::IPPROTO_TCP, netc::TCP_NODELAY, value)
    }

    pub fn nodelay(&self) -> io::Result<bool> {
        let raw: i32 = getsockopt(self, netc::IPPROTO_TCP, netc::TCP_NODELAY)?;
        Ok(raw != 0)
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        let mut nonblocking: i32 = if nonblocking { 1 } else { 0 };
        cvt(unsafe {
            netc::ioctl(
                self.as_raw_fd(),
                netc::FIONBIO,
                (&raw mut nonblocking) as *mut core::ffi::c_void,
            )
        })
        .map(drop)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        unimplemented!()
    }

    // This is used by sys_common code to abstract over Windows and Unix.
    pub fn as_raw(&self) -> RawFd {
        self.0.as_raw_fd()
    }
}

impl AsInner<FileDesc> for Socket {
    #[inline]
    fn as_inner(&self) -> &FileDesc {
        &self.0
    }
}

impl IntoInner<FileDesc> for Socket {
    fn into_inner(self) -> FileDesc {
        self.0
    }
}

impl FromInner<FileDesc> for Socket {
    fn from_inner(file_desc: FileDesc) -> Self {
        Self(file_desc)
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
