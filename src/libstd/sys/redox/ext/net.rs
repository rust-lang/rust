#![stable(feature = "unix_socket_redox", since = "1.29.0")]

//! Unix-specific networking functionality

use crate::fmt;
use crate::io::{self, Error, ErrorKind, Initializer};
use crate::net::Shutdown;
use crate::os::unix::io::{RawFd, AsRawFd, FromRawFd, IntoRawFd};
use crate::path::Path;
use crate::time::Duration;
use crate::sys::{cvt, fd::FileDesc, syscall};

/// An address associated with a Unix socket.
///
/// # Examples
///
/// ```
/// use std::os::unix::net::UnixListener;
///
/// let socket = match UnixListener::bind("/tmp/sock") {
///     Ok(sock) => sock,
///     Err(e) => {
///         println!("Couldn't bind: {:?}", e);
///         return
///     }
/// };
/// let addr = socket.local_addr().expect("Couldn't get local address");
/// ```
#[derive(Clone)]
#[stable(feature = "unix_socket_redox", since = "1.29.0")]
pub struct SocketAddr(());

impl SocketAddr {
    /// Returns the contents of this address if it is a `pathname` address.
    ///
    /// # Examples
    ///
    /// With a pathname:
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixListener;
    /// use std::path::Path;
    ///
    /// let socket = UnixListener::bind("/tmp/sock").unwrap();
    /// let addr = socket.local_addr().expect("Couldn't get local address");
    /// assert_eq!(addr.as_pathname(), Some(Path::new("/tmp/sock")));
    /// ```
    ///
    /// Without a pathname:
    ///
    /// ```
    /// use std::os::unix::net::UnixDatagram;
    ///
    /// let socket = UnixDatagram::unbound().unwrap();
    /// let addr = socket.local_addr().expect("Couldn't get local address");
    /// assert_eq!(addr.as_pathname(), None);
    /// ```
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
    pub fn as_pathname(&self) -> Option<&Path> {
        None
    }

    /// Returns `true` if the address is unnamed.
    ///
    /// # Examples
    ///
    /// A named address:
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixListener;
    ///
    /// let socket = UnixListener::bind("/tmp/sock").unwrap();
    /// let addr = socket.local_addr().expect("Couldn't get local address");
    /// assert_eq!(addr.is_unnamed(), false);
    /// ```
    ///
    /// An unnamed address:
    ///
    /// ```
    /// use std::os::unix::net::UnixDatagram;
    ///
    /// let socket = UnixDatagram::unbound().unwrap();
    /// let addr = socket.local_addr().expect("Couldn't get local address");
    /// assert_eq!(addr.is_unnamed(), true);
    /// ```
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
    pub fn is_unnamed(&self) -> bool {
        false
    }
}
#[stable(feature = "unix_socket_redox", since = "1.29.0")]
impl fmt::Debug for SocketAddr {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "SocketAddr")
    }
}

/// A Unix stream socket.
///
/// # Examples
///
/// ```no_run
/// use std::os::unix::net::UnixStream;
/// use std::io::prelude::*;
///
/// let mut stream = UnixStream::connect("/path/to/my/socket").unwrap();
/// stream.write_all(b"hello world").unwrap();
/// let mut response = String::new();
/// stream.read_to_string(&mut response).unwrap();
/// println!("{}", response);
/// ```
#[stable(feature = "unix_socket_redox", since = "1.29.0")]
pub struct UnixStream(FileDesc);

#[stable(feature = "unix_socket_redox", since = "1.29.0")]
impl fmt::Debug for UnixStream {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
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
    ///         println!("Couldn't connect: {:?}", e);
    ///         return
    ///     }
    /// };
    /// ```
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
    pub fn connect<P: AsRef<Path>>(path: P) -> io::Result<UnixStream> {
        if let Some(s) = path.as_ref().to_str() {
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
    ///         println!("Couldn't create a pair of sockets: {:?}", e);
    ///         return
    ///     }
    /// };
    /// ```
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
    pub fn pair() -> io::Result<(UnixStream, UnixStream)> {
        let server = cvt(syscall::open("chan:", syscall::O_CREAT | syscall::O_CLOEXEC))
            .map(FileDesc::new)?;
        let client = server.duplicate_path(b"connect")?;
        let stream = server.duplicate_path(b"listen")?;
        Ok((UnixStream(client), UnixStream(stream)))
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
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// let sock_copy = socket.try_clone().expect("Couldn't clone socket");
    /// ```
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
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
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// let addr = socket.local_addr().expect("Couldn't get local address");
    /// ```
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        Err(Error::new(ErrorKind::Other, "UnixStream::local_addr unimplemented on redox"))
    }

    /// Returns the socket address of the remote half of this connection.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    ///
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// let addr = socket.peer_addr().expect("Couldn't get peer address");
    /// ```
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        Err(Error::new(ErrorKind::Other, "UnixStream::peer_addr unimplemented on redox"))
    }

    /// Sets the read timeout for the socket.
    ///
    /// If the provided value is [`None`], then [`read`] calls will block
    /// indefinitely. An [`Err`] is returned if the zero [`Duration`] is passed to this
    /// method.
    ///
    /// [`None`]: ../../../../std/option/enum.Option.html#variant.None
    /// [`Err`]: ../../../../std/result/enum.Result.html#variant.Err
    /// [`read`]: ../../../../std/io/trait.Read.html#tymethod.read
    /// [`Duration`]: ../../../../std/time/struct.Duration.html
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    /// use std::time::Duration;
    ///
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// socket.set_read_timeout(Some(Duration::new(1, 0))).expect("Couldn't set read timeout");
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
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// let result = socket.set_read_timeout(Some(Duration::new(0, 0)));
    /// let err = result.unwrap_err();
    /// assert_eq!(err.kind(), io::ErrorKind::InvalidInput)
    /// ```
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
    pub fn set_read_timeout(&self, _timeout: Option<Duration>) -> io::Result<()> {
        Err(Error::new(ErrorKind::Other, "UnixStream::set_read_timeout unimplemented on redox"))
    }

    /// Sets the write timeout for the socket.
    ///
    /// If the provided value is [`None`], then [`write`] calls will block
    /// indefinitely. An [`Err`] is returned if the zero [`Duration`] is
    /// passed to this method.
    ///
    /// [`None`]: ../../../../std/option/enum.Option.html#variant.None
    /// [`Err`]: ../../../../std/result/enum.Result.html#variant.Err
    /// [`write`]: ../../../../std/io/trait.Write.html#tymethod.write
    /// [`Duration`]: ../../../../std/time/struct.Duration.html
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    /// use std::time::Duration;
    ///
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// socket.set_write_timeout(Some(Duration::new(1, 0))).expect("Couldn't set write timeout");
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
    /// let socket = UdpSocket::bind("127.0.0.1:34254").unwrap();
    /// let result = socket.set_write_timeout(Some(Duration::new(0, 0)));
    /// let err = result.unwrap_err();
    /// assert_eq!(err.kind(), io::ErrorKind::InvalidInput)
    /// ```
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
    pub fn set_write_timeout(&self, _timeout: Option<Duration>) -> io::Result<()> {
        Err(Error::new(ErrorKind::Other, "UnixStream::set_write_timeout unimplemented on redox"))
    }

    /// Returns the read timeout of this socket.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    /// use std::time::Duration;
    ///
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// socket.set_read_timeout(Some(Duration::new(1, 0))).expect("Couldn't set read timeout");
    /// assert_eq!(socket.read_timeout().unwrap(), Some(Duration::new(1, 0)));
    /// ```
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        Err(Error::new(ErrorKind::Other, "UnixStream::read_timeout unimplemented on redox"))
    }

    /// Returns the write timeout of this socket.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    /// use std::time::Duration;
    ///
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// socket.set_write_timeout(Some(Duration::new(1, 0))).expect("Couldn't set write timeout");
    /// assert_eq!(socket.write_timeout().unwrap(), Some(Duration::new(1, 0)));
    /// ```
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        Err(Error::new(ErrorKind::Other, "UnixStream::write_timeout unimplemented on redox"))
    }

    /// Moves the socket into or out of nonblocking mode.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    ///
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// socket.set_nonblocking(true).expect("Couldn't set nonblocking");
    /// ```
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.0.set_nonblocking(nonblocking)
    }

    /// Returns the value of the `SO_ERROR` option.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    ///
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// if let Ok(Some(err)) = socket.take_error() {
    ///     println!("Got error: {:?}", err);
    /// }
    /// ```
    ///
    /// # Platform specific
    /// On Redox this always returns `None`.
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        Ok(None)
    }

    /// Shuts down the read, write, or both halves of this connection.
    ///
    /// This function will cause all pending and future I/O calls on the
    /// specified portions to immediately return with an appropriate value
    /// (see the documentation of [`Shutdown`]).
    ///
    /// [`Shutdown`]: ../../../../std/net/enum.Shutdown.html
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixStream;
    /// use std::net::Shutdown;
    ///
    /// let socket = UnixStream::connect("/tmp/sock").unwrap();
    /// socket.shutdown(Shutdown::Both).expect("shutdown function failed");
    /// ```
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
    pub fn shutdown(&self, _how: Shutdown) -> io::Result<()> {
        Err(Error::new(ErrorKind::Other, "UnixStream::shutdown unimplemented on redox"))
    }
}

#[stable(feature = "unix_socket_redox", since = "1.29.0")]
impl io::Read for UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        io::Read::read(&mut &*self, buf)
    }

    #[inline]
    unsafe fn initializer(&self) -> Initializer {
        Initializer::nop()
    }
}

#[stable(feature = "unix_socket_redox", since = "1.29.0")]
impl<'a> io::Read for &'a UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }

    #[inline]
    unsafe fn initializer(&self) -> Initializer {
        Initializer::nop()
    }
}

#[stable(feature = "unix_socket_redox", since = "1.29.0")]
impl io::Write for UnixStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        io::Write::write(&mut &*self, buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        io::Write::flush(&mut &*self)
    }
}

#[stable(feature = "unix_socket_redox", since = "1.29.0")]
impl<'a> io::Write for &'a UnixStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[stable(feature = "unix_socket_redox", since = "1.29.0")]
impl AsRawFd for UnixStream {
    fn as_raw_fd(&self) -> RawFd {
        self.0.raw()
    }
}

#[stable(feature = "unix_socket_redox", since = "1.29.0")]
impl FromRawFd for UnixStream {
    unsafe fn from_raw_fd(fd: RawFd) -> UnixStream {
        UnixStream(FileDesc::new(fd))
    }
}

#[stable(feature = "unix_socket_redox", since = "1.29.0")]
impl IntoRawFd for UnixStream {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_raw()
    }
}

/// A structure representing a Unix domain socket server.
///
/// # Examples
///
/// ```no_run
/// use std::thread;
/// use std::os::unix::net::{UnixStream, UnixListener};
///
/// fn handle_client(stream: UnixStream) {
///     // ...
/// }
///
/// let listener = UnixListener::bind("/path/to/the/socket").unwrap();
///
/// // accept connections and process them, spawning a new thread for each one
/// for stream in listener.incoming() {
///     match stream {
///         Ok(stream) => {
///             /* connection succeeded */
///             thread::spawn(|| handle_client(stream));
///         }
///         Err(err) => {
///             /* connection failed */
///             break;
///         }
///     }
/// }
/// ```
#[stable(feature = "unix_socket_redox", since = "1.29.0")]
pub struct UnixListener(FileDesc);

#[stable(feature = "unix_socket_redox", since = "1.29.0")]
impl fmt::Debug for UnixListener {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut builder = fmt.debug_struct("UnixListener");
        builder.field("fd", &self.0.raw());
        if let Ok(addr) = self.local_addr() {
            builder.field("local", &addr);
        }
        builder.finish()
    }
}

impl UnixListener {
    /// Creates a new `UnixListener` bound to the specified socket.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixListener;
    ///
    /// let listener = match UnixListener::bind("/path/to/the/socket") {
    ///     Ok(sock) => sock,
    ///     Err(e) => {
    ///         println!("Couldn't connect: {:?}", e);
    ///         return
    ///     }
    /// };
    /// ```
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
    pub fn bind<P: AsRef<Path>>(path: P) -> io::Result<UnixListener> {
        if let Some(s) = path.as_ref().to_str() {
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

    /// Accepts a new incoming connection to this listener.
    ///
    /// This function will block the calling thread until a new Unix connection
    /// is established. When established, the corresponding [`UnixStream`] and
    /// the remote peer's address will be returned.
    ///
    /// [`UnixStream`]: ../../../../std/os/unix/net/struct.UnixStream.html
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixListener;
    ///
    /// let listener = UnixListener::bind("/path/to/the/socket").unwrap();
    ///
    /// match listener.accept() {
    ///     Ok((socket, addr)) => println!("Got a client: {:?}", addr),
    ///     Err(e) => println!("accept function failed: {:?}", e),
    /// }
    /// ```
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
    pub fn accept(&self) -> io::Result<(UnixStream, SocketAddr)> {
        self.0.duplicate_path(b"listen").map(|fd| (UnixStream(fd), SocketAddr(())))
    }

    /// Creates a new independently owned handle to the underlying socket.
    ///
    /// The returned `UnixListener` is a reference to the same socket that this
    /// object references. Both handles can be used to accept incoming
    /// connections and options set on one listener will affect the other.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixListener;
    ///
    /// let listener = UnixListener::bind("/path/to/the/socket").unwrap();
    ///
    /// let listener_copy = listener.try_clone().expect("try_clone failed");
    /// ```
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
    pub fn try_clone(&self) -> io::Result<UnixListener> {
        self.0.duplicate().map(UnixListener)
    }

    /// Returns the local socket address of this listener.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixListener;
    ///
    /// let listener = UnixListener::bind("/path/to/the/socket").unwrap();
    ///
    /// let addr = listener.local_addr().expect("Couldn't get local address");
    /// ```
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        Err(Error::new(ErrorKind::Other, "UnixListener::local_addr unimplemented on redox"))
    }

    /// Moves the socket into or out of nonblocking mode.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixListener;
    ///
    /// let listener = UnixListener::bind("/path/to/the/socket").unwrap();
    ///
    /// listener.set_nonblocking(true).expect("Couldn't set non blocking");
    /// ```
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.0.set_nonblocking(nonblocking)
    }

    /// Returns the value of the `SO_ERROR` option.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::os::unix::net::UnixListener;
    ///
    /// let listener = UnixListener::bind("/tmp/sock").unwrap();
    ///
    /// if let Ok(Some(err)) = listener.take_error() {
    ///     println!("Got error: {:?}", err);
    /// }
    /// ```
    ///
    /// # Platform specific
    /// On Redox this always returns `None`.
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        Ok(None)
    }

    /// Returns an iterator over incoming connections.
    ///
    /// The iterator will never return [`None`] and will also not yield the
    /// peer's [`SocketAddr`] structure.
    ///
    /// [`None`]: ../../../../std/option/enum.Option.html#variant.None
    /// [`SocketAddr`]: struct.SocketAddr.html
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::thread;
    /// use std::os::unix::net::{UnixStream, UnixListener};
    ///
    /// fn handle_client(stream: UnixStream) {
    ///     // ...
    /// }
    ///
    /// let listener = UnixListener::bind("/path/to/the/socket").unwrap();
    ///
    /// for stream in listener.incoming() {
    ///     match stream {
    ///         Ok(stream) => {
    ///             thread::spawn(|| handle_client(stream));
    ///         }
    ///         Err(err) => {
    ///             break;
    ///         }
    ///     }
    /// }
    /// ```
    #[stable(feature = "unix_socket_redox", since = "1.29.0")]
    pub fn incoming(&self) -> Incoming<'_> {
        Incoming { listener: self }
    }
}

#[stable(feature = "unix_socket_redox", since = "1.29.0")]
impl AsRawFd for UnixListener {
    fn as_raw_fd(&self) -> RawFd {
        self.0.raw()
    }
}

#[stable(feature = "unix_socket_redox", since = "1.29.0")]
impl FromRawFd for UnixListener {
    unsafe fn from_raw_fd(fd: RawFd) -> UnixListener {
        UnixListener(FileDesc::new(fd))
    }
}

#[stable(feature = "unix_socket_redox", since = "1.29.0")]
impl IntoRawFd for UnixListener {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_raw()
    }
}

#[stable(feature = "unix_socket_redox", since = "1.29.0")]
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
/// [`None`]: ../../../../std/option/enum.Option.html#variant.None
/// [`UnixListener`]: struct.UnixListener.html
///
/// # Examples
///
/// ```no_run
/// use std::thread;
/// use std::os::unix::net::{UnixStream, UnixListener};
///
/// fn handle_client(stream: UnixStream) {
///     // ...
/// }
///
/// let listener = UnixListener::bind("/path/to/the/socket").unwrap();
///
/// for stream in listener.incoming() {
///     match stream {
///         Ok(stream) => {
///             thread::spawn(|| handle_client(stream));
///         }
///         Err(err) => {
///             break;
///         }
///     }
/// }
/// ```
#[derive(Debug)]
#[stable(feature = "unix_socket_redox", since = "1.29.0")]
pub struct Incoming<'a> {
    listener: &'a UnixListener,
}

#[stable(feature = "unix_socket_redox", since = "1.29.0")]
impl<'a> Iterator for Incoming<'a> {
    type Item = io::Result<UnixStream>;

    fn next(&mut self) -> Option<io::Result<UnixStream>> {
        Some(self.listener.accept().map(|s| s.0))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::max_value(), None)
    }
}
