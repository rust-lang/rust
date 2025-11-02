use libc::{MSG_PEEK, c_int, c_void, size_t, sockaddr, socklen_t};

#[cfg(not(any(target_os = "espidf", target_os = "nuttx")))]
use crate::ffi::CStr;
use crate::io::{self, BorrowedBuf, BorrowedCursor, IoSlice, IoSliceMut};
use crate::net::{Shutdown, SocketAddr};
use crate::os::unix::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, RawFd};
use crate::sys::fd::FileDesc;
use crate::sys::net::{getsockopt, setsockopt};
use crate::sys::pal::IsMinusOne;
use crate::sys_common::{AsInner, FromInner, IntoInner};
use crate::time::{Duration, Instant};
use crate::{cmp, mem};

cfg_select! {
    target_vendor = "apple" => {
        use libc::SO_LINGER_SEC as SO_LINGER;
    }
    _ => {
        use libc::SO_LINGER;
    }
}

pub(super) use libc as netc;

use super::{socket_addr_from_c, socket_addr_to_c};
pub use crate::sys::{cvt, cvt_r};

#[expect(non_camel_case_types)]
pub type wrlen_t = size_t;

pub struct Socket(FileDesc);

pub fn init() {}

pub fn cvt_gai(err: c_int) -> io::Result<()> {
    if err == 0 {
        return Ok(());
    }

    // We may need to trigger a glibc workaround. See on_resolver_failure() for details.
    on_resolver_failure();

    #[cfg(not(any(target_os = "espidf", target_os = "nuttx")))]
    if err == libc::EAI_SYSTEM {
        return Err(io::Error::last_os_error());
    }

    #[cfg(not(any(target_os = "espidf", target_os = "nuttx")))]
    let detail = unsafe {
        // We can't always expect a UTF-8 environment. When we don't get that luxury,
        // it's better to give a low-quality error message than none at all.
        CStr::from_ptr(libc::gai_strerror(err)).to_string_lossy()
    };

    #[cfg(any(target_os = "espidf", target_os = "nuttx"))]
    let detail = "";

    Err(io::Error::new(
        io::ErrorKind::Uncategorized,
        &format!("failed to lookup address information: {detail}")[..],
    ))
}

impl Socket {
    pub fn new(family: c_int, ty: c_int) -> io::Result<Socket> {
        cfg_select! {
            any(
                target_os = "android",
                target_os = "dragonfly",
                target_os = "freebsd",
                target_os = "illumos",
                target_os = "hurd",
                target_os = "linux",
                target_os = "netbsd",
                target_os = "openbsd",
                target_os = "cygwin",
                target_os = "nto",
                target_os = "solaris",
            ) => {
                // On platforms that support it we pass the SOCK_CLOEXEC
                // flag to atomically create the socket and set it as
                // CLOEXEC. On Linux this was added in 2.6.27.
                let fd = cvt(unsafe { libc::socket(family, ty | libc::SOCK_CLOEXEC, 0) })?;
                let socket = Socket(unsafe { FileDesc::from_raw_fd(fd) });

                // DragonFlyBSD, FreeBSD and NetBSD use `SO_NOSIGPIPE` as a `setsockopt`
                // flag to disable `SIGPIPE` emission on socket.
                #[cfg(any(target_os = "freebsd", target_os = "netbsd", target_os = "dragonfly"))]
                unsafe { setsockopt(&socket, libc::SOL_SOCKET, libc::SO_NOSIGPIPE, 1)? };

                Ok(socket)
            }
            _ => {
                let fd = cvt(unsafe { libc::socket(family, ty, 0) })?;
                let fd = unsafe { FileDesc::from_raw_fd(fd) };
                fd.set_cloexec()?;
                let socket = Socket(fd);

                // macOS and iOS use `SO_NOSIGPIPE` as a `setsockopt`
                // flag to disable `SIGPIPE` emission on socket.
                #[cfg(target_vendor = "apple")]
                unsafe { setsockopt(&socket, libc::SOL_SOCKET, libc::SO_NOSIGPIPE, 1)? };

                Ok(socket)
            }
        }
    }

    #[cfg(not(target_os = "vxworks"))]
    pub fn new_pair(fam: c_int, ty: c_int) -> io::Result<(Socket, Socket)> {
        unsafe {
            let mut fds = [0, 0];

            cfg_select! {
                any(
                    target_os = "android",
                    target_os = "dragonfly",
                    target_os = "freebsd",
                    target_os = "illumos",
                    target_os = "linux",
                    target_os = "hurd",
                    target_os = "netbsd",
                    target_os = "openbsd",
                    target_os = "cygwin",
                    target_os = "nto",
                ) => {
                    // Like above, set cloexec atomically
                    cvt(libc::socketpair(fam, ty | libc::SOCK_CLOEXEC, 0, fds.as_mut_ptr()))?;
                    Ok((Socket(FileDesc::from_raw_fd(fds[0])), Socket(FileDesc::from_raw_fd(fds[1]))))
                }
                _ => {
                    cvt(libc::socketpair(fam, ty, 0, fds.as_mut_ptr()))?;
                    let a = FileDesc::from_raw_fd(fds[0]);
                    let b = FileDesc::from_raw_fd(fds[1]);
                    a.set_cloexec()?;
                    b.set_cloexec()?;
                    Ok((Socket(a), Socket(b)))
                }
            }
        }
    }

    #[cfg(target_os = "vxworks")]
    pub fn new_pair(_fam: c_int, _ty: c_int) -> io::Result<(Socket, Socket)> {
        unimplemented!()
    }

    pub fn connect(&self, addr: &SocketAddr) -> io::Result<()> {
        let (addr, len) = socket_addr_to_c(addr);
        loop {
            let result = unsafe { libc::connect(self.as_raw_fd(), addr.as_ptr(), len) };
            if result.is_minus_one() {
                let err = crate::sys::os::errno();
                match err {
                    libc::EINTR => continue,
                    libc::EISCONN => return Ok(()),
                    _ => return Err(io::Error::from_raw_os_error(err)),
                }
            }
            return Ok(());
        }
    }

    pub fn connect_timeout(&self, addr: &SocketAddr, timeout: Duration) -> io::Result<()> {
        self.set_nonblocking(true)?;
        let r = unsafe {
            let (addr, len) = socket_addr_to_c(addr);
            cvt(libc::connect(self.as_raw_fd(), addr.as_ptr(), len))
        };
        self.set_nonblocking(false)?;

        match r {
            Ok(_) => return Ok(()),
            // there's no ErrorKind for EINPROGRESS :(
            Err(ref e) if e.raw_os_error() == Some(libc::EINPROGRESS) => {}
            Err(e) => return Err(e),
        }

        let mut pollfd = libc::pollfd { fd: self.as_raw_fd(), events: libc::POLLOUT, revents: 0 };

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

            match unsafe { libc::poll(&mut pollfd, 1, timeout) } {
                -1 => {
                    let err = io::Error::last_os_error();
                    if !err.is_interrupted() {
                        return Err(err);
                    }
                }
                0 => {}
                _ => {
                    if cfg!(target_os = "vxworks") {
                        // VxWorks poll does not return  POLLHUP or POLLERR in revents. Check if the
                        // connection actually succeeded and return ok only when the socket is
                        // ready and no errors were found.
                        if let Some(e) = self.take_error()? {
                            return Err(e);
                        }
                    } else {
                        // linux returns POLLOUT|POLLERR|POLLHUP for refused connections (!), so look
                        // for POLLHUP or POLLERR rather than read readiness
                        if pollfd.revents & (libc::POLLHUP | libc::POLLERR) != 0 {
                            let e = self.take_error()?.unwrap_or_else(|| {
                                io::const_error!(
                                    io::ErrorKind::Uncategorized,
                                    "no error set after POLLHUP",
                                )
                            });
                            return Err(e);
                        }
                    }

                    return Ok(());
                }
            }
        }
    }

    pub fn accept(&self, storage: *mut sockaddr, len: *mut socklen_t) -> io::Result<Socket> {
        // Unfortunately the only known way right now to accept a socket and
        // atomically set the CLOEXEC flag is to use the `accept4` syscall on
        // platforms that support it. On Linux, this was added in 2.6.28,
        // glibc 2.10 and musl 0.9.5.
        cfg_select! {
            any(
                target_os = "android",
                target_os = "dragonfly",
                target_os = "freebsd",
                target_os = "illumos",
                target_os = "linux",
                target_os = "hurd",
                target_os = "netbsd",
                target_os = "openbsd",
                target_os = "cygwin",
            ) => {
                unsafe {
                    let fd = cvt_r(|| libc::accept4(self.as_raw_fd(), storage, len, libc::SOCK_CLOEXEC))?;
                    Ok(Socket(FileDesc::from_raw_fd(fd)))
                }
            }
            _ => {
                unsafe {
                    let fd = cvt_r(|| libc::accept(self.as_raw_fd(), storage, len))?;
                    let fd = FileDesc::from_raw_fd(fd);
                    fd.set_cloexec()?;
                    Ok(Socket(fd))
                }
            }
        }
    }

    pub fn duplicate(&self) -> io::Result<Socket> {
        self.0.duplicate().map(Socket)
    }

    pub fn send_with_flags(&self, buf: &[u8], flags: c_int) -> io::Result<usize> {
        let len = cmp::min(buf.len(), <wrlen_t>::MAX as usize) as wrlen_t;
        let ret = cvt(unsafe {
            libc::send(self.as_raw_fd(), buf.as_ptr() as *const c_void, len, flags)
        })?;
        Ok(ret as usize)
    }

    fn recv_with_flags(&self, mut buf: BorrowedCursor<'_>, flags: c_int) -> io::Result<()> {
        let ret = cvt(unsafe {
            libc::recv(
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
        self.recv_with_flags(buf.unfilled(), MSG_PEEK)?;
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

    fn recv_from_with_flags(
        &self,
        buf: &mut [u8],
        flags: c_int,
    ) -> io::Result<(usize, SocketAddr)> {
        // The `recvfrom` function will fill in the storage with the address,
        // so we don't need to zero it here.
        // reference: https://linux.die.net/man/2/recvfrom
        let mut storage: mem::MaybeUninit<libc::sockaddr_storage> = mem::MaybeUninit::uninit();
        let mut addrlen = size_of_val(&storage) as libc::socklen_t;

        let n = cvt(unsafe {
            libc::recvfrom(
                self.as_raw_fd(),
                buf.as_mut_ptr() as *mut c_void,
                buf.len(),
                flags,
                (&raw mut storage) as *mut _,
                &mut addrlen,
            )
        })?;
        Ok((n as usize, unsafe { socket_addr_from_c(storage.as_ptr(), addrlen as usize)? }))
    }

    pub fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.recv_from_with_flags(buf, 0)
    }

    #[cfg(any(target_os = "android", target_os = "linux", target_os = "cygwin"))]
    pub fn recv_msg(&self, msg: &mut libc::msghdr) -> io::Result<usize> {
        let n = cvt(unsafe { libc::recvmsg(self.as_raw_fd(), msg, libc::MSG_CMSG_CLOEXEC) })?;
        Ok(n as usize)
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

    #[cfg(any(target_os = "android", target_os = "linux", target_os = "cygwin"))]
    pub fn send_msg(&self, msg: &mut libc::msghdr) -> io::Result<usize> {
        let n = cvt(unsafe { libc::sendmsg(self.as_raw_fd(), msg, 0) })?;
        Ok(n as usize)
    }

    pub fn set_timeout(&self, dur: Option<Duration>, kind: libc::c_int) -> io::Result<()> {
        let timeout = match dur {
            Some(dur) => {
                if dur.as_secs() == 0 && dur.subsec_nanos() == 0 {
                    return Err(io::Error::ZERO_TIMEOUT);
                }

                let secs = if dur.as_secs() > libc::time_t::MAX as u64 {
                    libc::time_t::MAX
                } else {
                    dur.as_secs() as libc::time_t
                };
                let mut timeout = libc::timeval {
                    tv_sec: secs,
                    tv_usec: dur.subsec_micros() as libc::suseconds_t,
                };
                if timeout.tv_sec == 0 && timeout.tv_usec == 0 {
                    timeout.tv_usec = 1;
                }
                timeout
            }
            None => libc::timeval { tv_sec: 0, tv_usec: 0 },
        };
        unsafe { setsockopt(self, libc::SOL_SOCKET, kind, timeout) }
    }

    pub fn timeout(&self, kind: libc::c_int) -> io::Result<Option<Duration>> {
        let raw: libc::timeval = unsafe { getsockopt(self, libc::SOL_SOCKET, kind)? };
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
            Shutdown::Write => libc::SHUT_WR,
            Shutdown::Read => libc::SHUT_RD,
            Shutdown::Both => libc::SHUT_RDWR,
        };
        cvt(unsafe { libc::shutdown(self.as_raw_fd(), how) })?;
        Ok(())
    }

    #[cfg(not(target_os = "cygwin"))]
    pub fn set_linger(&self, linger: Option<Duration>) -> io::Result<()> {
        let linger = libc::linger {
            l_onoff: linger.is_some() as libc::c_int,
            l_linger: linger.unwrap_or_default().as_secs() as libc::c_int,
        };

        unsafe { setsockopt(self, libc::SOL_SOCKET, SO_LINGER, linger) }
    }

    #[cfg(target_os = "cygwin")]
    pub fn set_linger(&self, linger: Option<Duration>) -> io::Result<()> {
        let linger = libc::linger {
            l_onoff: linger.is_some() as libc::c_ushort,
            l_linger: linger.unwrap_or_default().as_secs() as libc::c_ushort,
        };

        unsafe { setsockopt(self, libc::SOL_SOCKET, SO_LINGER, linger) }
    }

    pub fn linger(&self) -> io::Result<Option<Duration>> {
        let val: libc::linger = unsafe { getsockopt(self, libc::SOL_SOCKET, SO_LINGER)? };

        Ok((val.l_onoff != 0).then(|| Duration::from_secs(val.l_linger as u64)))
    }

    pub fn set_nodelay(&self, nodelay: bool) -> io::Result<()> {
        unsafe { setsockopt(self, libc::IPPROTO_TCP, libc::TCP_NODELAY, nodelay as c_int) }
    }

    pub fn nodelay(&self) -> io::Result<bool> {
        let raw: c_int = unsafe { getsockopt(self, libc::IPPROTO_TCP, libc::TCP_NODELAY)? };
        Ok(raw != 0)
    }

    #[cfg(any(target_os = "android", target_os = "linux", target_os = "cygwin"))]
    pub fn set_quickack(&self, quickack: bool) -> io::Result<()> {
        unsafe { setsockopt(self, libc::IPPROTO_TCP, libc::TCP_QUICKACK, quickack as c_int) }
    }

    #[cfg(any(target_os = "android", target_os = "linux", target_os = "cygwin"))]
    pub fn quickack(&self) -> io::Result<bool> {
        let raw: c_int = unsafe { getsockopt(self, libc::IPPROTO_TCP, libc::TCP_QUICKACK)? };
        Ok(raw != 0)
    }

    // bionic libc makes no use of this flag
    #[cfg(target_os = "linux")]
    pub fn set_deferaccept(&self, accept: Duration) -> io::Result<()> {
        let val = cmp::min(accept.as_secs(), c_int::MAX as u64) as c_int;
        unsafe { setsockopt(self, libc::IPPROTO_TCP, libc::TCP_DEFER_ACCEPT, val) }
    }

    #[cfg(target_os = "linux")]
    pub fn deferaccept(&self) -> io::Result<Duration> {
        let raw: c_int = unsafe { getsockopt(self, libc::IPPROTO_TCP, libc::TCP_DEFER_ACCEPT)? };
        Ok(Duration::from_secs(raw as _))
    }

    #[cfg(any(target_os = "freebsd", target_os = "netbsd"))]
    pub fn set_acceptfilter(&self, name: &CStr) -> io::Result<()> {
        if !name.to_bytes().is_empty() {
            const AF_NAME_MAX: usize = 16;
            let mut buf = [0; AF_NAME_MAX];
            for (src, dst) in name.to_bytes().iter().zip(&mut buf[..AF_NAME_MAX - 1]) {
                *dst = *src as libc::c_char;
            }
            let mut arg: libc::accept_filter_arg = unsafe { mem::zeroed() };
            arg.af_name = buf;
            unsafe { setsockopt(self, libc::SOL_SOCKET, libc::SO_ACCEPTFILTER, &mut arg) }
        } else {
            unsafe {
                setsockopt(
                    self,
                    libc::SOL_SOCKET,
                    libc::SO_ACCEPTFILTER,
                    core::ptr::null_mut() as *mut c_void,
                )
            }
        }
    }

    #[cfg(any(target_os = "freebsd", target_os = "netbsd"))]
    pub fn acceptfilter(&self) -> io::Result<&CStr> {
        let arg: libc::accept_filter_arg =
            unsafe { getsockopt(self, libc::SOL_SOCKET, libc::SO_ACCEPTFILTER)? };
        let s: &[u8] =
            unsafe { core::slice::from_raw_parts(arg.af_name.as_ptr() as *const u8, 16) };
        let name = CStr::from_bytes_with_nul(s).unwrap();
        Ok(name)
    }

    #[cfg(any(target_os = "solaris", target_os = "illumos"))]
    pub fn set_exclbind(&self, excl: bool) -> io::Result<()> {
        // not yet on libc crate
        const SO_EXCLBIND: i32 = 0x1015;
        unsafe { setsockopt(self, libc::SOL_SOCKET, SO_EXCLBIND, excl) }
    }

    #[cfg(any(target_os = "solaris", target_os = "illumos"))]
    pub fn exclbind(&self) -> io::Result<bool> {
        // not yet on libc crate
        const SO_EXCLBIND: i32 = 0x1015;
        let raw: c_int = unsafe { getsockopt(self, libc::SOL_SOCKET, SO_EXCLBIND)? };
        Ok(raw != 0)
    }

    #[cfg(any(target_os = "android", target_os = "linux", target_os = "cygwin"))]
    pub fn set_passcred(&self, passcred: bool) -> io::Result<()> {
        unsafe { setsockopt(self, libc::SOL_SOCKET, libc::SO_PASSCRED, passcred as libc::c_int) }
    }

    #[cfg(any(target_os = "android", target_os = "linux", target_os = "cygwin"))]
    pub fn passcred(&self) -> io::Result<bool> {
        let passcred: libc::c_int =
            unsafe { getsockopt(self, libc::SOL_SOCKET, libc::SO_PASSCRED)? };
        Ok(passcred != 0)
    }

    #[cfg(target_os = "netbsd")]
    pub fn set_local_creds(&self, local_creds: bool) -> io::Result<()> {
        unsafe { setsockopt(self, 0 as libc::c_int, libc::LOCAL_CREDS, local_creds as libc::c_int) }
    }

    #[cfg(target_os = "netbsd")]
    pub fn local_creds(&self) -> io::Result<bool> {
        let local_creds: libc::c_int =
            unsafe { getsockopt(self, 0 as libc::c_int, libc::LOCAL_CREDS)? };
        Ok(local_creds != 0)
    }

    #[cfg(target_os = "freebsd")]
    pub fn set_local_creds_persistent(&self, local_creds_persistent: bool) -> io::Result<()> {
        unsafe {
            setsockopt(
                self,
                libc::AF_LOCAL,
                libc::LOCAL_CREDS_PERSISTENT,
                local_creds_persistent as libc::c_int,
            )
        }
    }

    #[cfg(target_os = "freebsd")]
    pub fn local_creds_persistent(&self) -> io::Result<bool> {
        let local_creds_persistent: libc::c_int =
            unsafe { getsockopt(self, libc::AF_LOCAL, libc::LOCAL_CREDS_PERSISTENT)? };
        Ok(local_creds_persistent != 0)
    }

    #[cfg(not(any(target_os = "solaris", target_os = "illumos", target_os = "vita")))]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        let mut nonblocking = nonblocking as libc::c_int;
        cvt(unsafe { libc::ioctl(self.as_raw_fd(), libc::FIONBIO, &mut nonblocking) }).map(drop)
    }

    #[cfg(target_os = "vita")]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        let option = nonblocking as libc::c_int;
        unsafe { setsockopt(self, libc::SOL_SOCKET, libc::SO_NONBLOCK, option) }
    }

    #[cfg(any(target_os = "solaris", target_os = "illumos"))]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        // FIONBIO is inadequate for sockets on illumos/Solaris, so use the
        // fcntl(F_[GS]ETFL)-based method provided by FileDesc instead.
        self.0.set_nonblocking(nonblocking)
    }

    #[cfg(any(target_os = "linux", target_os = "freebsd", target_os = "openbsd"))]
    pub fn set_mark(&self, mark: u32) -> io::Result<()> {
        #[cfg(target_os = "linux")]
        let option = libc::SO_MARK;
        #[cfg(target_os = "freebsd")]
        let option = libc::SO_USER_COOKIE;
        #[cfg(target_os = "openbsd")]
        let option = libc::SO_RTABLE;
        unsafe { setsockopt(self, libc::SOL_SOCKET, option, mark as libc::c_int) }
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        let raw: c_int = unsafe { getsockopt(self, libc::SOL_SOCKET, libc::SO_ERROR)? };
        if raw == 0 { Ok(None) } else { Ok(Some(io::Error::from_raw_os_error(raw as i32))) }
    }

    // This is used by sys_common code to abstract over Windows and Unix.
    pub fn as_raw(&self) -> RawFd {
        self.as_raw_fd()
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

impl IntoRawFd for Socket {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_raw_fd()
    }
}

impl FromRawFd for Socket {
    unsafe fn from_raw_fd(raw_fd: RawFd) -> Self {
        Self(FromRawFd::from_raw_fd(raw_fd))
    }
}

// In versions of glibc prior to 2.26, there's a bug where the DNS resolver
// will cache the contents of /etc/resolv.conf, so changes to that file on disk
// can be ignored by a long-running program. That can break DNS lookups on e.g.
// laptops where the network comes and goes. See
// https://sourceware.org/bugzilla/show_bug.cgi?id=984. Note however that some
// distros including Debian have patched glibc to fix this for a long time.
//
// A workaround for this bug is to call the res_init libc function, to clear
// the cached configs. Unfortunately, while we believe glibc's implementation
// of res_init is thread-safe, we know that other implementations are not
// (https://github.com/rust-lang/rust/issues/43592). Code here in std could
// try to synchronize its res_init calls with a Mutex, but that wouldn't
// protect programs that call into libc in other ways. So instead of calling
// res_init unconditionally, we call it only when we detect we're linking
// against glibc version < 2.26. (That is, when we both know its needed and
// believe it's thread-safe).
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn on_resolver_failure() {
    use crate::sys;

    // If the version fails to parse, we treat it the same as "not glibc".
    if let Some(version) = sys::os::glibc_version() {
        if version < (2, 26) {
            unsafe { libc::res_init() };
        }
    }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn on_resolver_failure() {}
