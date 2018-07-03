// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use fmt;
use io::{self, Error, ErrorKind, Initializer};
use net::Shutdown;
use os::unix::io::{RawFd, AsRawFd, FromRawFd, IntoRawFd};
use path::Path;
use time::Duration;
use sys::{cvt, fd::FileDesc, syscall};

#[stable(feature = "unix_socket", since = "1.10.0")]
#[derive(Clone)]
pub struct SocketAddr(());

impl SocketAddr {
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn is_unnamed(&self) -> bool {
        false
    }
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn as_pathname(&self) -> Option<&Path> {
        None
    }
}
#[stable(feature = "unix_socket", since = "1.10.0")]
impl fmt::Debug for SocketAddr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "SocketAddr")
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
pub struct UnixStream(FileDesc);

#[stable(feature = "unix_socket", since = "1.10.0")]
impl fmt::Debug for UnixStream {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut builder = fmt.debug_struct("UnixStream");
        builder.field("fd", &self.0.raw());
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
        if let Some(s) = path.to_str() {
            cvt(syscall::open(format!("chan:{}", s), syscall::O_CLOEXEC))
                .map(FileDesc::new)
                .map(UnixStream)
        } else {
            Err(Error::new(
                ErrorKind::Other,
                "UnixStream::connect: non-utf8 paths not supported on redox"
            ))
        }
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn pair() -> io::Result<(UnixStream, UnixStream)> {
        let server = cvt(syscall::open("chan:", syscall::O_CREAT | syscall::O_CLOEXEC))
            .map(FileDesc::new)?;
        let client = server.duplicate_path(b"connect")?;
        let stream = server.duplicate_path(b"listen")?;
        Ok((UnixStream(client), UnixStream(stream)))
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn try_clone(&self) -> io::Result<UnixStream> {
        self.0.duplicate().map(UnixStream)
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        Err(Error::new(ErrorKind::Other, "UnixStream::local_addr unimplemented on redox"))
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        Err(Error::new(ErrorKind::Other, "UnixStream::peer_addr unimplemented on redox"))
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_read_timeout(&self, _timeout: Option<Duration>) -> io::Result<()> {
        Err(Error::new(ErrorKind::Other, "UnixStream::set_read_timeout unimplemented on redox"))
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_write_timeout(&self, _timeout: Option<Duration>) -> io::Result<()> {
        Err(Error::new(ErrorKind::Other, "UnixStream::set_write_timeout unimplemented on redox"))
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        Err(Error::new(ErrorKind::Other, "UnixStream::read_timeout unimplemented on redox"))
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        Err(Error::new(ErrorKind::Other, "UnixStream::write_timeout unimplemented on redox"))
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.0.set_nonblocking(nonblocking)
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        Ok(None)
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn shutdown(&self, _how: Shutdown) -> io::Result<()> {
        Err(Error::new(ErrorKind::Other, "UnixStream::shutdown unimplemented on redox"))
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

#[stable(feature = "unix_socket", since = "1.10.0")]
impl AsRawFd for UnixStream {
    fn as_raw_fd(&self) -> RawFd {
        self.0.raw()
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl FromRawFd for UnixStream {
    unsafe fn from_raw_fd(fd: RawFd) -> UnixStream {
        UnixStream(FileDesc::new(fd))
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl IntoRawFd for UnixStream {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_raw()
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
pub struct UnixListener(FileDesc);

#[stable(feature = "unix_socket", since = "1.10.0")]
impl fmt::Debug for UnixListener {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut builder = fmt.debug_struct("UnixListener");
        builder.field("fd", &self.0.raw());
        if let Ok(addr) = self.local_addr() {
            builder.field("local", &addr);
        }
        builder.finish()
    }
}

impl UnixListener {
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn bind(path: &Path) -> io::Result<UnixListener> {
        if let Some(s) = path.to_str() {
            cvt(syscall::open(format!("chan:{}", s), syscall::O_CREAT | syscall::O_CLOEXEC))
                .map(FileDesc::new)
                .map(UnixListener)
        } else {
            Err(Error::new(
                ErrorKind::Other,
                "UnixListener::bind: non-utf8 paths not supported on redox"
            ))
        }
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn accept(&self) -> io::Result<(UnixStream, SocketAddr)> {
        self.0.duplicate_path(b"listen").map(|fd| (UnixStream(fd), SocketAddr(())))
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn try_clone(&self) -> io::Result<UnixListener> {
        self.0.duplicate().map(UnixListener)
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        Err(Error::new(ErrorKind::Other, "UnixListener::local_addr unimplemented on redox"))
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.0.set_nonblocking(nonblocking)
    }

    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        Ok(None)
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl AsRawFd for UnixListener {
    fn as_raw_fd(&self) -> RawFd {
        self.0.raw()
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl FromRawFd for UnixListener {
    unsafe fn from_raw_fd(fd: RawFd) -> UnixListener {
        UnixListener(FileDesc::new(fd))
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl IntoRawFd for UnixListener {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_raw()
    }
}
