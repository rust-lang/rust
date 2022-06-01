//! WASI-specific networking functionality

#![unstable(feature = "wasi_ext", issue = "71213")]
#![allow(unused_variables, dead_code)]

use crate::io;
use crate::fmt;
use crate::path::Path;
use super::{SocketAddr, UnixStream};
use crate::os::wasi::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use crate::sys::net::Socket;
use crate::sys_common::{AsInner, FromInner, IntoInner};

/// A structure representing a Unix domain socket server.
///
/// Not currently supported on this platform
#[stable(feature = "unix_socket", since = "1.10.0")]
pub struct UnixListener(Socket);

#[stable(feature = "unix_socket", since = "1.10.0")]
impl fmt::Debug for UnixListener {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut builder = fmt.debug_struct("UnixListener");
        builder.finish()
    }
}

impl UnixListener {
    /// Creates a new `UnixListener` bound to the specified socket.
    ///
    /// Not currently supported on this platform
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn bind<P: AsRef<Path>>(_path: P) -> io::Result<UnixListener> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ))
    }

    /// Creates a new `UnixListener` bound to the specified [`socket address`].
    ///
    /// [`socket address`]: crate::os::wasi::net::SocketAddr
    ///
    /// Not currently supported on this platform
    #[unstable(feature = "unix_socket_abstract", issue = "85410")]
    pub fn bind_addr(socket_addr: &SocketAddr) -> io::Result<UnixListener> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ))
    }

    /// Accepts a new incoming connection to this listener.
    ///
    /// Not currently supported on this platform
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn accept(&self) -> io::Result<(UnixStream, SocketAddr)> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ))
    }

    /// Accepts a new incoming connection to this listener (or times out).
    ///
    /// Not currently supported on this platform
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn accept_timeout(&self, _timeout: crate::time::Duration) -> io::Result<(UnixStream, SocketAddr)> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ))
    }

    /// Creates a new independently owned handle to the underlying socket.
    ///
    /// Not currently supported on this platform
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn try_clone(&self) -> io::Result<UnixListener> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ))
    }

    /// Returns the local socket address of this listener.
    ///
    /// Not currently supported on this platform
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ))
    }

    /// Moves the socket into or out of nonblocking mode.
    ///
    /// Not currently supported on this platform
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ))
    }

    /// Returns the value of the `SO_ERROR` option.
    ///
    /// Not currently supported on this platform
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ))
    }

    /// Returns an iterator over incoming connections.
    ///
    /// Not currently supported on this platform
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn incoming(&self) -> Incoming<'_> {
        Incoming { listener: self }
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl AsRawFd for UnixListener {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.0.as_inner().as_raw_fd()
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl FromRawFd for UnixListener {
    #[inline]
    unsafe fn from_raw_fd(fd: RawFd) -> UnixListener {
        unsafe {
            UnixListener(Socket::from_inner(FromInner::from_inner(OwnedFd::from_raw_fd(fd))))
        }
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl IntoRawFd for UnixListener {
    #[inline]
    fn into_raw_fd(self) -> RawFd {
        self.0.into_inner().into_inner().into_raw_fd()
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl AsFd for UnixListener {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_inner().as_fd()
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl From<OwnedFd> for UnixListener {
    #[inline]
    fn from(fd: OwnedFd) -> UnixListener {
        UnixListener(Socket::from_inner(FromInner::from_inner(fd)))
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl From<UnixListener> for OwnedFd {
    #[inline]
    fn from(listener: UnixListener) -> OwnedFd {
        listener.0.into_inner().into_inner()
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl<'a> IntoIterator for &'a UnixListener {
    type Item = io::Result<UnixStream>;
    type IntoIter = Incoming<'a>;

    fn into_iter(self) -> Incoming<'a> {
        self.incoming()
    }
}

/// An iterator over incoming connections to a [`UnixListener`].
///
/// It will never return [`None`].
///
/// Not currently supported on this platform
#[derive(Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "unix_socket", since = "1.10.0")]
pub struct Incoming<'a> {
    listener: &'a UnixListener,
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl<'a> Iterator for Incoming<'a> {
    type Item = io::Result<UnixStream>;

    fn next(&mut self) -> Option<io::Result<UnixStream>> {
        Some(self.listener.accept().map(|s| s.0))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }
}
