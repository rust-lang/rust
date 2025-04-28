use super::{SocketAddr, sockaddr_un};
#[cfg(any(doc, target_os = "android", target_os = "linux"))]
use super::{SocketAncillary, recv_vectored_with_ancillary_from, send_vectored_with_ancillary_to};
#[cfg(any(
    target_os = "android",
    target_os = "linux",
    target_os = "dragonfly",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "nto",
    target_vendor = "apple",
    target_os = "cygwin"
))]
use super::{UCred, peer_cred};
use crate::fmt;
use crate::io::{self, IoSlice, IoSliceMut};
use crate::net::Shutdown;
use crate::os::unix::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use crate::path::Path;
use crate::sealed::Sealed;
use crate::sys::cvt;
use crate::sys::net::Socket;
use crate::sys_common::{AsInner, FromInner};
use crate::time::Duration;

/// A Unix stream socket.
///
/// # Examples
///
/// ```no_run
/// use std::os::unix::net::UnixStream;
/// use std::io::prelude::*;
///
/// fn main() -> std::io::Result<()> {
///     let mut stream = UnixStream::connect("/path/to/my/socket")?;
///     stream.write_all(b"hello world")?;
///     let mut response = String::new();
///     stream.read_to_string(&mut response)?;
///     println!("{response}");
///     Ok(())
/// }
/// ```
#[stable(feature = "unix_socket", since = "1.10.0")]
pub struct UnixStream(pub(super) Socket);

/// Allows extension traits within `std`.
#[unstable(feature = "sealed", issue = "none")]
impl Sealed for UnixStream {}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl fmt::Debug for UnixStream {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
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
    /// Connects to the socket named by `path`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    ///
    /// let socket = match UnixStream::connect("/tmp/sock") {
    ///     Ok(sock) => sock,
    ///     Err(e) => {
    ///         println!("Couldn't connect: {e:?}");
    ///         return
    ///     }
    /// };
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn connect<P: AsRef<Path>>(path: P) -> io::Result<UnixStream> {
        unsafe {
            let inner = Socket::new_raw(libc::AF_UNIX, libc::SOCK_STREAM)?;
            let (addr, len) = sockaddr_un(path.as_ref())?;

            cvt(libc::connect(inner.as_raw_fd(), (&raw const addr) as *const _, len))?;
            Ok(UnixStream(inner))
        }
    }

    /// Connects to the socket specified by [`address`].
    ///
    /// [`address`]: crate::os::unix::net::SocketAddr
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::{UnixListener, UnixStream};
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let listener = UnixListener::bind("/path/to/the/socket")?;
    ///     let addr = listener.local_addr()?;
    ///
    ///     let sock = match UnixStream::connect_addr(&addr) {
    ///         Ok(sock) => sock,
    ///         Err(e) => {
    ///             println!("Couldn't connect: {e:?}");
    ///             return Err(e)
    ///         }
    ///     };
    ///     Ok(())
    /// }
    /// ````
    #[stable(feature = "unix_socket_abstract", since = "1.70.0")]
    pub fn connect_addr(socket_addr: &SocketAddr) -> io::Result<UnixStream> {
        unsafe {
            let inner = Socket::new_raw(libc::AF_UNIX, libc::SOCK_STREAM)?;
            cvt(libc::connect(
                inner.as_raw_fd(),
                (&raw const socket_addr.addr) as *const _,
                socket_addr.len,
            ))?;
            Ok(UnixStream(inner))
        }
    }

    /// Creates an unnamed pair of connected sockets.
    ///
    /// Returns two `UnixStream`s which are connected to each other.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    ///
    /// let (sock1, sock2) = match UnixStream::pair() {
    ///     Ok((sock1, sock2)) => (sock1, sock2),
    ///     Err(e) => {
    ///         println!("Couldn't create a pair of sockets: {e:?}");
    ///         return
    ///     }
    /// };
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn pair() -> io::Result<(UnixStream, UnixStream)> {
        let (i1, i2) = Socket::new_pair(libc::AF_UNIX, libc::SOCK_STREAM)?;
        Ok((UnixStream(i1), UnixStream(i2)))
    }

    /// Creates a new independently owned handle to the underlying socket.
    ///
    /// The returned `UnixStream` is a reference to the same stream that this
    /// object references. Both handles will read and write the same stream of
    /// data, and options set on one stream will be propagated to the other
    /// stream.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixStream::connect("/tmp/sock")?;
    ///     let sock_copy = socket.try_clone().expect("Couldn't clone socket");
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn try_clone(&self) -> io::Result<UnixStream> {
        self.0.duplicate().map(UnixStream)
    }

    /// Returns the socket address of the local half of this connection.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixStream::connect("/tmp/sock")?;
    ///     let addr = socket.local_addr().expect("Couldn't get local address");
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        SocketAddr::new(|addr, len| unsafe { libc::getsockname(self.as_raw_fd(), addr, len) })
    }

    /// Returns the socket address of the remote half of this connection.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixStream::connect("/tmp/sock")?;
    ///     let addr = socket.peer_addr().expect("Couldn't get peer address");
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        SocketAddr::new(|addr, len| unsafe { libc::getpeername(self.as_raw_fd(), addr, len) })
    }

    /// Gets the peer credentials for this Unix domain socket.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(peer_credentials_unix_socket)]
    /// use std::os::unix::net::UnixStream;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixStream::connect("/tmp/sock")?;
    ///     let peer_cred = socket.peer_cred().expect("Couldn't get peer credentials");
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "peer_credentials_unix_socket", issue = "42839", reason = "unstable")]
    #[cfg(any(
        target_os = "android",
        target_os = "linux",
        target_os = "dragonfly",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "nto",
        target_vendor = "apple",
        target_os = "cygwin"
    ))]
    pub fn peer_cred(&self) -> io::Result<UCred> {
        peer_cred(self)
    }

    /// Sets the read timeout for the socket.
    ///
    /// If the provided value is [`None`], then [`read`] calls will block
    /// indefinitely. An [`Err`] is returned if the zero [`Duration`] is passed to this
    /// method.
    ///
    /// [`read`]: io::Read::read
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    /// use std::time::Duration;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixStream::connect("/tmp/sock")?;
    ///     socket.set_read_timeout(Some(Duration::new(1, 0))).expect("Couldn't set read timeout");
    ///     Ok(())
    /// }
    /// ```
    ///
    /// An [`Err`] is returned if the zero [`Duration`] is passed to this
    /// method:
    ///
    /// ```no_run
    /// use std::io;
    /// use std::os::unix::net::UnixStream;
    /// use std::time::Duration;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixStream::connect("/tmp/sock")?;
    ///     let result = socket.set_read_timeout(Some(Duration::new(0, 0)));
    ///     let err = result.unwrap_err();
    ///     assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        self.0.set_timeout(timeout, libc::SO_RCVTIMEO)
    }

    /// Sets the write timeout for the socket.
    ///
    /// If the provided value is [`None`], then [`write`] calls will block
    /// indefinitely. An [`Err`] is returned if the zero [`Duration`] is
    /// passed to this method.
    ///
    /// [`read`]: io::Read::read
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    /// use std::time::Duration;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixStream::connect("/tmp/sock")?;
    ///     socket.set_write_timeout(Some(Duration::new(1, 0)))
    ///         .expect("Couldn't set write timeout");
    ///     Ok(())
    /// }
    /// ```
    ///
    /// An [`Err`] is returned if the zero [`Duration`] is passed to this
    /// method:
    ///
    /// ```no_run
    /// use std::io;
    /// use std::net::UdpSocket;
    /// use std::time::Duration;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UdpSocket::bind("127.0.0.1:34254")?;
    ///     let result = socket.set_write_timeout(Some(Duration::new(0, 0)));
    ///     let err = result.unwrap_err();
    ///     assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        self.0.set_timeout(timeout, libc::SO_SNDTIMEO)
    }

    /// Returns the read timeout of this socket.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    /// use std::time::Duration;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixStream::connect("/tmp/sock")?;
    ///     socket.set_read_timeout(Some(Duration::new(1, 0))).expect("Couldn't set read timeout");
    ///     assert_eq!(socket.read_timeout()?, Some(Duration::new(1, 0)));
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        self.0.timeout(libc::SO_RCVTIMEO)
    }

    /// Returns the write timeout of this socket.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    /// use std::time::Duration;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixStream::connect("/tmp/sock")?;
    ///     socket.set_write_timeout(Some(Duration::new(1, 0)))
    ///         .expect("Couldn't set write timeout");
    ///     assert_eq!(socket.write_timeout()?, Some(Duration::new(1, 0)));
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        self.0.timeout(libc::SO_SNDTIMEO)
    }

    /// Moves the socket into or out of nonblocking mode.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixStream::connect("/tmp/sock")?;
    ///     socket.set_nonblocking(true).expect("Couldn't set nonblocking");
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.0.set_nonblocking(nonblocking)
    }

    /// Set the id of the socket for network filtering purpose
    ///
    #[cfg_attr(
        any(target_os = "linux", target_os = "freebsd", target_os = "openbsd"),
        doc = "```no_run"
    )]
    #[cfg_attr(
        not(any(target_os = "linux", target_os = "freebsd", target_os = "openbsd")),
        doc = "```ignore"
    )]
    /// #![feature(unix_set_mark)]
    /// use std::os::unix::net::UnixStream;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let sock = UnixStream::connect("/tmp/sock")?;
    ///     sock.set_mark(32)?;
    ///     Ok(())
    /// }
    /// ```
    #[cfg(any(doc, target_os = "linux", target_os = "freebsd", target_os = "openbsd",))]
    #[unstable(feature = "unix_set_mark", issue = "96467")]
    pub fn set_mark(&self, mark: u32) -> io::Result<()> {
        self.0.set_mark(mark)
    }

    /// Returns the value of the `SO_ERROR` option.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixStream::connect("/tmp/sock")?;
    ///     if let Ok(Some(err)) = socket.take_error() {
    ///         println!("Got error: {err:?}");
    ///     }
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Platform specific
    /// On Redox this always returns `None`.
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        self.0.take_error()
    }

    /// Shuts down the read, write, or both halves of this connection.
    ///
    /// This function will cause all pending and future I/O calls on the
    /// specified portions to immediately return with an appropriate value
    /// (see the documentation of [`Shutdown`]).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    /// use std::net::Shutdown;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixStream::connect("/tmp/sock")?;
    ///     socket.shutdown(Shutdown::Both).expect("shutdown function failed");
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "unix_socket", since = "1.10.0")]
    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        self.0.shutdown(how)
    }

    /// Receives data on the socket from the remote address to which it is
    /// connected, without removing that data from the queue. On success,
    /// returns the number of bytes peeked.
    ///
    /// Successive calls return the same data. This is accomplished by passing
    /// `MSG_PEEK` as a flag to the underlying `recv` system call.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(unix_socket_peek)]
    ///
    /// use std::os::unix::net::UnixStream;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixStream::connect("/tmp/sock")?;
    ///     let mut buf = [0; 10];
    ///     let len = socket.peek(&mut buf).expect("peek failed");
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "unix_socket_peek", issue = "76923")]
    pub fn peek(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.peek(buf)
    }

    /// Receives data and ancillary data from socket.
    ///
    /// On success, returns the number of bytes read.
    ///
    /// # Examples
    ///
    #[cfg_attr(any(target_os = "android", target_os = "linux"), doc = "```no_run")]
    #[cfg_attr(not(any(target_os = "android", target_os = "linux")), doc = "```ignore")]
    /// #![feature(unix_socket_ancillary_data)]
    /// use std::os::unix::net::{UnixStream, SocketAncillary, AncillaryData};
    /// use std::io::IoSliceMut;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixStream::connect("/tmp/sock")?;
    ///     let mut buf1 = [1; 8];
    ///     let mut buf2 = [2; 16];
    ///     let mut buf3 = [3; 8];
    ///     let mut bufs = &mut [
    ///         IoSliceMut::new(&mut buf1),
    ///         IoSliceMut::new(&mut buf2),
    ///         IoSliceMut::new(&mut buf3),
    ///     ][..];
    ///     let mut fds = [0; 8];
    ///     let mut ancillary_buffer = [0; 128];
    ///     let mut ancillary = SocketAncillary::new(&mut ancillary_buffer[..]);
    ///     let size = socket.recv_vectored_with_ancillary(bufs, &mut ancillary)?;
    ///     println!("received {size}");
    ///     for ancillary_result in ancillary.messages() {
    ///         if let AncillaryData::ScmRights(scm_rights) = ancillary_result.unwrap() {
    ///             for fd in scm_rights {
    ///                 println!("receive file descriptor: {fd}");
    ///             }
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[cfg(any(doc, target_os = "android", target_os = "linux"))]
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn recv_vectored_with_ancillary(
        &self,
        bufs: &mut [IoSliceMut<'_>],
        ancillary: &mut SocketAncillary<'_>,
    ) -> io::Result<usize> {
        let (count, _, _) = recv_vectored_with_ancillary_from(&self.0, bufs, ancillary)?;

        Ok(count)
    }

    /// Sends data and ancillary data on the socket.
    ///
    /// On success, returns the number of bytes written.
    ///
    /// # Examples
    ///
    #[cfg_attr(any(target_os = "android", target_os = "linux"), doc = "```no_run")]
    #[cfg_attr(not(any(target_os = "android", target_os = "linux")), doc = "```ignore")]
    /// #![feature(unix_socket_ancillary_data)]
    /// use std::os::unix::net::{UnixStream, SocketAncillary};
    /// use std::io::IoSlice;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UnixStream::connect("/tmp/sock")?;
    ///     let buf1 = [1; 8];
    ///     let buf2 = [2; 16];
    ///     let buf3 = [3; 8];
    ///     let bufs = &[
    ///         IoSlice::new(&buf1),
    ///         IoSlice::new(&buf2),
    ///         IoSlice::new(&buf3),
    ///     ][..];
    ///     let fds = [0, 1, 2];
    ///     let mut ancillary_buffer = [0; 128];
    ///     let mut ancillary = SocketAncillary::new(&mut ancillary_buffer[..]);
    ///     ancillary.add_fds(&fds[..]);
    ///     socket.send_vectored_with_ancillary(bufs, &mut ancillary)
    ///         .expect("send_vectored_with_ancillary function failed");
    ///     Ok(())
    /// }
    /// ```
    #[cfg(any(doc, target_os = "android", target_os = "linux"))]
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn send_vectored_with_ancillary(
        &self,
        bufs: &[IoSlice<'_>],
        ancillary: &mut SocketAncillary<'_>,
    ) -> io::Result<usize> {
        send_vectored_with_ancillary_to(&self.0, None, bufs, ancillary)
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl io::Read for UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        io::Read::read(&mut &*self, buf)
    }

    fn read_buf(&mut self, buf: io::BorrowedCursor<'_>) -> io::Result<()> {
        io::Read::read_buf(&mut &*self, buf)
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

    fn read_buf(&mut self, buf: io::BorrowedCursor<'_>) -> io::Result<()> {
        self.0.read_buf(buf)
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

    #[inline]
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
        UnixStream(Socket::from_inner(FromInner::from_inner(OwnedFd::from_raw_fd(fd))))
    }
}

#[stable(feature = "unix_socket", since = "1.10.0")]
impl IntoRawFd for UnixStream {
    #[inline]
    fn into_raw_fd(self) -> RawFd {
        self.0.into_raw_fd()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsFd for UnixStream {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<UnixStream> for OwnedFd {
    /// Takes ownership of a [`UnixStream`]'s socket file descriptor.
    #[inline]
    fn from(unix_stream: UnixStream) -> OwnedFd {
        unsafe { OwnedFd::from_raw_fd(unix_stream.into_raw_fd()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<OwnedFd> for UnixStream {
    #[inline]
    fn from(owned: OwnedFd) -> Self {
        unsafe { Self::from_raw_fd(owned.into_raw_fd()) }
    }
}

impl AsInner<Socket> for UnixStream {
    #[inline]
    fn as_inner(&self) -> &Socket {
        &self.0
    }
}
