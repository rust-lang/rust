// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unix-specific networking functionality

#[cfg(unix)]
use libc;

// FIXME(#43348): Make libc adapt #[doc(cfg(...))] so we don't need these fake definitions here?
#[cfg(not(unix))]
mod libc {
    pub use libc::c_int;
    pub type socklen_t = u32;
    pub struct sockaddr;
    #[derive(Clone)]
    pub struct sockaddr_un;
}

use ascii;
use ffi::OsStr;
use fmt;
use io::{self, Initializer};
use mem;
use net::Shutdown;
use os::unix::ffi::OsStrExt;
use os::unix::io::{RawFd, AsRawFd, FromRawFd, IntoRawFd};
use path::Path;
use time::Duration;
use sys::cvt;
use sys::net::Socket;
use sys::ext::net::*;
use sys_common::{AsInner, FromInner, IntoInner};

enum AddressKind<'a> {
    Unnamed,
    Pathname(&'a Path),
    Abstract(&'a [u8]),
}

#[stable(feature = "unix_socket", since = "1.10.0")]
#[derive(Clone)]
pub struct SocketAddr {
    addr: libc::sockaddr_un,
    len: libc::socklen_t,
}

impl SocketAddr {
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn new<F>(f: F) -> io::Result<SocketAddr>
        where F: FnOnce(*mut libc::sockaddr, *mut libc::socklen_t) -> libc::c_int
    {
        unsafe {
            let mut addr: libc::sockaddr_un = mem::zeroed();
            let mut len = mem::size_of::<libc::sockaddr_un>() as libc::socklen_t;
            cvt(f(&mut addr as *mut _ as *mut _, &mut len))?;
            SocketAddr::from_parts(addr, len)
        }
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn from_parts(addr: libc::sockaddr_un, mut len: libc::socklen_t)
        -> io::Result<SocketAddr>
    {
        if len == 0 {
            // When there is a datagram from unnamed unix socket
            // linux returns zero bytes of address
            len = sun_path_offset() as libc::socklen_t;  // i.e. zero-length address
        } else if addr.sun_family != libc::AF_UNIX as libc::sa_family_t {
            return Err(io::Error::new(io::ErrorKind::InvalidInput,
                                      "file descriptor did not correspond to a Unix socket"));
        }

        Ok(SocketAddr {
            addr,
            len,
        })
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn is_unnamed(&self) -> bool {
        if let AddressKind::Unnamed = self.address() {
            true
        } else {
            false
        }
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn as_pathname(&self) -> Option<&Path> {
        if let AddressKind::Pathname(path) = self.address() {
            Some(path)
        } else {
            None
        }
    }

    fn address<'a>(&'a self) -> AddressKind<'a> {
        let len = self.len as usize - sun_path_offset();
        let path = unsafe { mem::transmute::<&[libc::c_char], &[u8]>(&self.addr.sun_path) };

        // macOS seems to return a len of 16 and a zeroed sun_path for unnamed addresses
        if len == 0
            || (cfg!(not(any(target_os = "linux", target_os = "android")))
                && self.addr.sun_path[0] == 0)
        {
            AddressKind::Unnamed
        } else if self.addr.sun_path[0] == 0 {
            AddressKind::Abstract(&path[1..len])
        } else {
            AddressKind::Pathname(OsStr::from_bytes(&path[..len - 1]).as_ref())
        }
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl fmt::Debug for SocketAddr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self.address() {
            AddressKind::Unnamed => write!(fmt, "(unnamed)"),
            AddressKind::Abstract(name) => write!(fmt, "{} (abstract)", AsciiEscaped(name)),
            AddressKind::Pathname(path) => write!(fmt, "{:?} (pathname)", path),
        }
    }
}

struct AsciiEscaped<'a>(&'a [u8]);

impl<'a> fmt::Display for AsciiEscaped<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "\"")?;
        for byte in self.0.iter().cloned().flat_map(ascii::escape_default) {
            write!(fmt, "{}", byte as char)?;
        }
        write!(fmt, "\"")
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
pub struct UnixStream(Socket);

#[stable(feature = "unix_socket", since = "1.10.0")]
impl fmt::Debug for UnixStream {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut builder = fmt.debug_struct("UnixStream");
        builder.field("fd", self.0.as_inner());
        if let Ok(addr) = self.local_addr() {
            builder.field("local", &addr);
        }
        if let Ok(addr) = self.peer_addr() {
            builder.field("peer", &addr);
        }
        builder.finish()
    }
}

impl UnixStream {
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn connect(path: &Path) -> io::Result<UnixStream> {
        unsafe {
            let inner = Socket::new_raw(libc::AF_UNIX, libc::SOCK_STREAM)?;
            let (addr, len) = sockaddr_un(path)?;

            cvt(libc::connect(*inner.as_inner(), &addr as *const _ as *const _, len))?;
            Ok(UnixStream(inner))
        }
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn pair() -> io::Result<(UnixStream, UnixStream)> {
        let (i1, i2) = Socket::new_pair(libc::AF_UNIX, libc::SOCK_STREAM)?;
        Ok((UnixStream(i1), UnixStream(i2)))
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn try_clone(&self) -> io::Result<UnixStream> {
        self.0.duplicate().map(UnixStream)
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        SocketAddr::new(|addr, len| unsafe { libc::getsockname(*self.0.as_inner(), addr, len) })
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        SocketAddr::new(|addr, len| unsafe { libc::getpeername(*self.0.as_inner(), addr, len) })
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        self.0.set_timeout(timeout, libc::SO_RCVTIMEO)
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        self.0.set_timeout(timeout, libc::SO_SNDTIMEO)
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        self.0.timeout(libc::SO_RCVTIMEO)
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        self.0.timeout(libc::SO_SNDTIMEO)
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.0.set_nonblocking(nonblocking)
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        self.0.take_error()
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        self.0.shutdown(how)
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl<'a> io::Read for &'a UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }

    #[inline]
    unsafe fn initializer(&self) -> Initializer {
        Initializer::nop()
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl<'a> io::Write for &'a UnixStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl AsRawFd for UnixStream {
    fn as_raw_fd(&self) -> RawFd {
        *self.0.as_inner()
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl FromRawFd for UnixStream {
    unsafe fn from_raw_fd(fd: RawFd) -> UnixStream {
        UnixStream(Socket::from_inner(fd))
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for UnixStream {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_inner()
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
pub struct UnixListener(Socket);

#[stable(feature = "unix_socket", since = "1.10.0")]
impl fmt::Debug for UnixListener {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut builder = fmt.debug_struct("UnixListener");
        builder.field("fd", self.0.as_inner());
        if let Ok(addr) = self.local_addr() {
            builder.field("local", &addr);
        }
        builder.finish()
    }
}

impl UnixListener {
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn bind(path: &Path) -> io::Result<UnixListener> {
        unsafe {
            let inner = Socket::new_raw(libc::AF_UNIX, libc::SOCK_STREAM)?;
            let (addr, len) = sockaddr_un(path)?;

            cvt(libc::bind(*inner.as_inner(), &addr as *const _ as *const _, len as _))?;
            cvt(libc::listen(*inner.as_inner(), 128))?;

            Ok(UnixListener(inner))
        }
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn accept(&self) -> io::Result<(UnixStream, SocketAddr)> {
        let mut storage: libc::sockaddr_un = unsafe { mem::zeroed() };
        let mut len = mem::size_of_val(&storage) as libc::socklen_t;
        let sock = self.0.accept(&mut storage as *mut _ as *mut _, &mut len)?;
        let addr = SocketAddr::from_parts(storage, len)?;
        Ok((UnixStream(sock), addr))
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn try_clone(&self) -> io::Result<UnixListener> {
        self.0.duplicate().map(UnixListener)
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        SocketAddr::new(|addr, len| unsafe { libc::getsockname(*self.0.as_inner(), addr, len) })
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.0.set_nonblocking(nonblocking)
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        self.0.take_error()
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl AsRawFd for UnixListener {
    fn as_raw_fd(&self) -> RawFd {
        *self.0.as_inner()
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl FromRawFd for UnixListener {
    unsafe fn from_raw_fd(fd: RawFd) -> UnixListener {
        UnixListener(Socket::from_inner(fd))
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl IntoRawFd for UnixListener {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_inner()
    }
}
