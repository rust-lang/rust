#![allow(dead_code)]

use crate::cmp;
use crate::io::{self, IoSlice, IoSliceMut};
use crate::mem;
use crate::net::{Shutdown, SocketAddr};
use crate::os::hermit::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, RawFd};
use crate::sys::hermit::fd::FileDesc;
use crate::sys::time::Instant;
use crate::sys_common::net::{getsockopt, setsockopt, sockaddr_to_addr};
use crate::sys_common::{AsInner, FromInner, IntoInner};
use crate::time::Duration;

use core::ffi::c_int;

#[allow(unused_extern_crates)]
pub extern crate hermit_abi as netc;

pub use crate::sys::{cvt, cvt_r};

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

/// Checks whether the HermitCore's socket interface has been started already, and
/// if not, starts it.
pub fn init() {
    if unsafe { netc::network_init() } < 0 {
        panic!("Unable to initialize network interface");
    }
}

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

    pub fn connect_timeout(&self, addr: &SocketAddr, timeout: Duration) -> io::Result<()> {
        self.set_nonblocking(true)?;
        let r = unsafe {
            let (addr, len) = addr.into_inner();
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
            return Err(io::const_io_error!(
                io::ErrorKind::InvalidInput,
                "cannot set a 0 duration timeout",
            ));
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
                    if err.kind() != io::ErrorKind::Interrupted {
                        return Err(err);
                    }
                }
                0 => {}
                _ => {
                    // linux returns POLLOUT|POLLERR|POLLHUP for refused connections (!), so look
                    // for POLLHUP rather than read readiness
                    if pollfd.revents & netc::POLLHUP != 0 {
                        let e = self.take_error()?.unwrap_or_else(|| {
                            io::const_io_error!(
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

    fn recv_with_flags(&self, buf: &mut [u8], flags: i32) -> io::Result<usize> {
        let ret =
            cvt(unsafe { netc::recv(self.0.as_raw_fd(), buf.as_mut_ptr(), buf.len(), flags) })?;
        Ok(ret as usize)
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.recv_with_flags(buf, 0)
    }

    pub fn peek(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.recv_with_flags(buf, netc::MSG_PEEK)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let mut size: isize = 0;

        for i in bufs.iter_mut() {
            let ret: isize =
                cvt(unsafe { netc::read(self.0.as_raw_fd(), i.as_mut_ptr(), i.len()) })?;

            if ret != 0 {
                size += ret;
            }
        }

        Ok(size.try_into().unwrap())
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        true
    }

    fn recv_from_with_flags(&self, buf: &mut [u8], flags: i32) -> io::Result<(usize, SocketAddr)> {
        let mut storage: netc::sockaddr_storage = unsafe { mem::zeroed() };
        let mut addrlen = mem::size_of_val(&storage) as netc::socklen_t;

        let n = cvt(unsafe {
            netc::recvfrom(
                self.as_raw_fd(),
                buf.as_mut_ptr(),
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
        self.recv_from_with_flags(buf, netc::MSG_PEEK)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let sz = cvt(unsafe { netc::write(self.0.as_raw_fd(), buf.as_ptr(), buf.len()) })?;
        Ok(sz.try_into().unwrap())
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let mut size: isize = 0;

        for i in bufs.iter() {
            size += cvt(unsafe { netc::write(self.0.as_raw_fd(), i.as_ptr(), i.len()) })?;
        }

        Ok(size.try_into().unwrap())
    }

    pub fn is_write_vectored(&self) -> bool {
        true
    }

    pub fn set_timeout(&self, dur: Option<Duration>, kind: i32) -> io::Result<()> {
        let timeout = match dur {
            Some(dur) => {
                if dur.as_secs() == 0 && dur.subsec_nanos() == 0 {
                    return Err(io::const_io_error!(
                        io::ErrorKind::InvalidInput,
                        "cannot set a 0 duration timeout",
                    ));
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
        cvt(unsafe { netc::shutdown_socket(self.as_raw_fd(), how) })?;
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
                &mut nonblocking as *mut _ as *mut core::ffi::c_void,
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
    fn as_raw_fd(&self) -> RawFd {
        self.0.as_raw_fd()
    }
}
