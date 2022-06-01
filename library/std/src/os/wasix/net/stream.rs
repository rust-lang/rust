#![allow(unused_variables, dead_code)]
use super::{SocketAddr};
use crate::fmt;
use crate::io::{self, IoSlice, IoSliceMut};
use crate::net::Shutdown;
use crate::os::unix::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use crate::path::Path;
use crate::sys::net::Socket;
use crate::sys_common::{FromInner};
use crate::time::Duration;

/// A Unix stream socket.
///
/// Not supported on this platform
#[stable(feature = "unix_socket", since = "1.10.0")]
pub struct UnixStream(pub(super) Socket);

#[stable(feature = "unix_socket", since = "1.10.0")]
impl fmt::Debug for UnixStream {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut builder = fmt.debug_struct("UnixStream");
        builder.finish()
    }
}

impl UnixStream {
    /// Connects to the socket named by `path`.
    ///
    /// Not currently supported on this platform
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn connect<P: AsRef<Path>>(path: P) -> io::Result<UnixStream> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ))
    }

    /// Connects to the socket specified by [`address`].
    ///
    /// Not currently supported on this platform
    #[unstable(feature = "unix_socket_abstract", issue = "85410")]
    pub fn connect_addr(socket_addr: &SocketAddr) -> io::Result<UnixStream> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ))
    }

    /// Creates an unnamed pair of connected sockets.
    ///
    /// Not currently supported on this platform
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn pair() -> io::Result<(UnixStream, UnixStream)> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ))
    }

    /// Creates a new independently owned handle to the underlying socket.
    ///
    /// Not currently supported on this platform
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn try_clone(&self) -> io::Result<UnixStream> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ))
    }

    /// Returns the socket address of the local half of this connection.
    ///
    /// Not currently supported on this platform
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ))
    }

    /// Returns the socket address of the remote half of this connection.
    ///
    /// Not currently supported on this platform
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ))
    }

    /// Sets the read timeout for the socket.
    ///
    /// Not currently supported on this platform
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ))
    }

    /// Sets the write timeout for the socket.
    ///
    /// Not currently supported on this platform
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ))
    }

    /// Returns the read timeout of this socket.
    ///
    /// Not currently supported on this platforn
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ))
    }

    /// Returns the write timeout of this socket.
    ///
    /// Not currently supported on this platform
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
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

    /// Shuts down the read, write, or both halves of this connection.
    ///
    /// Not currently supported on this platform
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ))
    }

    /// Receives data on the socket from the remote address to which it is
    /// connected, without removing that data from the queue. On success,
    /// returns the number of bytes peeked.
    ///
    /// Not currently supported on this platform
    #[unstable(feature = "unix_socket_peek", issue = "76923")]
    pub fn peek(&self, buf: &mut [u8]) -> io::Result<usize> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "unix sockets are not supported on this platform",
        ))
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl io::Read for UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        io::Read::read(&mut &*self, buf)
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        io::Read::read_vectored(&mut &*self, bufs)
    }

    #[inline]
    fn is_read_vectored(&self) -> bool {
        io::Read::is_read_vectored(&&*self)
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl<'a> io::Read for &'a UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.0.read_vectored(bufs)
    }

    #[inline]
    fn is_read_vectored(&self) -> bool {
        self.0.is_read_vectored()
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl io::Write for UnixStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        io::Write::write(&mut &*self, buf)
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        io::Write::write_vectored(&mut &*self, bufs)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        io::Write::is_write_vectored(&&*self)
    }

    fn flush(&mut self) -> io::Result<()> {
        io::Write::flush(&mut &*self)
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl<'a> io::Write for &'a UnixStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.0.write_vectored(bufs)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        self.0.is_write_vectored()
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl AsRawFd for UnixStream {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.0.as_raw_fd()
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl FromRawFd for UnixStream {
    #[inline]
    unsafe fn from_raw_fd(fd: RawFd) -> UnixStream {
        unsafe {
            UnixStream(Socket::from_inner(FromInner::from_inner(OwnedFd::from_raw_fd(fd))))
        }
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl IntoRawFd for UnixStream {
    #[inline]
    fn into_raw_fd(self) -> RawFd {
        self.0.into_raw_fd()
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl AsFd for UnixStream {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl From<UnixStream> for OwnedFd {
    #[inline]
    fn from(unix_stream: UnixStream) -> OwnedFd {
        unsafe { OwnedFd::from_raw_fd(unix_stream.into_raw_fd()) }
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl From<OwnedFd> for UnixStream {
    #[inline]
    fn from(owned: OwnedFd) -> Self {
        unsafe { Self::from_raw_fd(owned.into_raw_fd()) }
    }
}
