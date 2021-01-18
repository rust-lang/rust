#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(all(test, not(target_os = "emscripten")))]
mod tests;

use crate::io::prelude::*;

use crate::fmt;
use crate::io::{self, IoSlice, IoSliceMut};
use crate::net::{Shutdown, SocketAddr, ToSocketAddrs};
use crate::sys_common::net as net_imp;
use crate::sys_common::{AsInner, FromInner, IntoInner};
use crate::time::Duration;

/// A TCP stream between a local and a remote socket.
///
/// After creating a `TcpStream` by either [`connect`]ing to a remote host or
/// [`accept`]ing a connection on a [`TcpListener`], data can be transmitted
/// by [reading] and [writing] to it.
///
/// The connection will be closed when the value is dropped. The reading and writing
/// portions of the connection can also be shut down individually with the [`shutdown`]
/// method.
///
/// The Transmission Control Protocol is specified in [IETF RFC 793].
///
/// [`accept`]: TcpListener::accept
/// [`connect`]: TcpStream::connect
/// [IETF RFC 793]: https://tools.ietf.org/html/rfc793
/// [reading]: Read
/// [`shutdown`]: TcpStream::shutdown
/// [writing]: Write
///
/// # Examples
///
/// ```no_run
/// use std::io::prelude::*;
/// use std::net::TcpStream;
///
/// fn main() -> std::io::Result<()> {
///     let mut stream = TcpStream::connect("127.0.0.1:34254")?;
///
///     stream.write(&[1])?;
///     stream.read(&mut [0; 128])?;
///     Ok(())
/// } // the stream is closed here
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct TcpStream(net_imp::TcpStream);

/// A TCP socket server, listening for connections.
///
/// After creating a `TcpListener` by [`bind`]ing it to a socket address, it listens
/// for incoming TCP connections. These can be accepted by calling [`accept`] or by
/// iterating over the [`Incoming`] iterator returned by [`incoming`][`TcpListener::incoming`].
///
/// The socket will be closed when the value is dropped.
///
/// The Transmission Control Protocol is specified in [IETF RFC 793].
///
/// [`accept`]: TcpListener::accept
/// [`bind`]: TcpListener::bind
/// [IETF RFC 793]: https://tools.ietf.org/html/rfc793
///
/// # Examples
///
/// ```no_run
/// use std::net::{TcpListener, TcpStream};
///
/// fn handle_client(stream: TcpStream) {
///     // ...
/// }
///
/// fn main() -> std::io::Result<()> {
///     let listener = TcpListener::bind("127.0.0.1:80")?;
///
///     // accept connections and process them serially
///     for stream in listener.incoming() {
///         handle_client(stream?);
///     }
///     Ok(())
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct TcpListener(net_imp::TcpListener);

/// An iterator that infinitely [`accept`]s connections on a [`TcpListener`].
///
/// This `struct` is created by the [`TcpListener::incoming`] method.
/// See its documentation for more.
///
/// [`accept`]: TcpListener::accept
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug)]
pub struct Incoming<'a> {
    listener: &'a TcpListener,
}

/// An iterator that infinitely [`accept`]s connections on a [`TcpListener`].
///
/// This `struct` is created by the [`TcpListener::into_incoming`] method.
/// See its documentation for more.
///
/// [`accept`]: TcpListener::accept
#[derive(Debug)]
#[unstable(feature = "tcplistener_into_incoming", issue = "88339")]
pub struct IntoIncoming {
    listener: TcpListener,
}

impl TcpStream {
    /// Opens a TCP connection to a remote host.
    ///
    /// `addr` is an address of the remote host. Anything which implements
    /// [`ToSocketAddrs`] trait can be supplied for the address; see this trait
    /// documentation for concrete examples.
    ///
    /// If `addr` yields multiple addresses, `connect` will be attempted with
    /// each of the addresses until a connection is successful. If none of
    /// the addresses result in a successful connection, the error returned from
    /// the last connection attempt (the last address) is returned.
    ///
    /// # Examples
    ///
    /// Open a TCP connection to `127.0.0.1:8080`:
    ///
    /// ```no_run
    /// use std::net::TcpStream;
    ///
    /// if let Ok(stream) = TcpStream::connect("127.0.0.1:8080") {
    ///     println!("Connected to the server!");
    /// } else {
    ///     println!("Couldn't connect to server...");
    /// }
    /// ```
    ///
    /// Open a TCP connection to `127.0.0.1:8080`. If the connection fails, open
    /// a TCP connection to `127.0.0.1:8081`:
    ///
    /// ```no_run
    /// use std::net::{SocketAddr, TcpStream};
    ///
    /// let addrs = [
    ///     SocketAddr::from(([127, 0, 0, 1], 8080)),
    ///     SocketAddr::from(([127, 0, 0, 1], 8081)),
    /// ];
    /// if let Ok(stream) = TcpStream::connect(&addrs[..]) {
    ///     println!("Connected to the server!");
    /// } else {
    ///     println!("Couldn't connect to server...");
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn connect<A: ToSocketAddrs>(addr: A) -> io::Result<TcpStream> {
        super::each_addr(addr, net_imp::TcpStream::connect).map(TcpStream)
    }

    /// Opens a TCP connection to a remote host with a timeout.
    ///
    /// Unlike `connect`, `connect_timeout` takes a single [`SocketAddr`] since
    /// timeout must be applied to individual addresses.
    ///
    /// It is an error to pass a zero `Duration` to this function.
    ///
    /// Unlike other methods on `TcpStream`, this does not correspond to a
    /// single system call. It instead calls `connect` in nonblocking mode and
    /// then uses an OS-specific mechanism to await the completion of the
    /// connection request.
    #[stable(feature = "tcpstream_connect_timeout", since = "1.21.0")]
    pub fn connect_timeout(addr: &SocketAddr, timeout: Duration) -> io::Result<TcpStream> {
        net_imp::TcpStream::connect_timeout(addr, timeout).map(TcpStream)
    }

    /// Returns the socket address of the remote peer of this TCP connection.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream};
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///                        .expect("Couldn't connect to the server...");
    /// assert_eq!(stream.peer_addr().unwrap(),
    ///            SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 8080)));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        self.0.peer_addr()
    }

    /// Returns the socket address of the local half of this TCP connection.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::{IpAddr, Ipv4Addr, TcpStream};
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///                        .expect("Couldn't connect to the server...");
    /// assert_eq!(stream.local_addr().unwrap().ip(),
    ///            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.0.socket_addr()
    }

    /// Shuts down the read, write, or both halves of this connection.
    ///
    /// This function will cause all pending and future I/O on the specified
    /// portions to return immediately with an appropriate value (see the
    /// documentation of [`Shutdown`]).
    ///
    /// # Platform-specific behavior
    ///
    /// Calling this function multiple times may result in different behavior,
    /// depending on the operating system. On Linux, the second call will
    /// return `Ok(())`, but on macOS, it will return `ErrorKind::NotConnected`.
    /// This may change in the future.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::{Shutdown, TcpStream};
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///                        .expect("Couldn't connect to the server...");
    /// stream.shutdown(Shutdown::Both).expect("shutdown call failed");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        self.0.shutdown(how)
    }

    /// Creates a new independently owned handle to the underlying socket.
    ///
    /// The returned `TcpStream` is a reference to the same stream that this
    /// object references. Both handles will read and write the same stream of
    /// data, and options set on one stream will be propagated to the other
    /// stream.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::TcpStream;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///                        .expect("Couldn't connect to the server...");
    /// let stream_clone = stream.try_clone().expect("clone failed...");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn try_clone(&self) -> io::Result<TcpStream> {
        self.0.duplicate().map(TcpStream)
    }

    /// Sets the read timeout to the timeout specified.
    ///
    /// If the value specified is [`None`], then [`read`] calls will block
    /// indefinitely. An [`Err`] is returned if the zero [`Duration`] is
    /// passed to this method.
    ///
    /// # Platform-specific behavior
    ///
    /// Platforms may return a different error code whenever a read times out as
    /// a result of setting this option. For example Unix typically returns an
    /// error of the kind [`WouldBlock`], but Windows may return [`TimedOut`].
    ///
    /// [`read`]: Read::read
    /// [`WouldBlock`]: io::ErrorKind::WouldBlock
    /// [`TimedOut`]: io::ErrorKind::TimedOut
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::TcpStream;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///                        .expect("Couldn't connect to the server...");
    /// stream.set_read_timeout(None).expect("set_read_timeout call failed");
    /// ```
    ///
    /// An [`Err`] is returned if the zero [`Duration`] is passed to this
    /// method:
    ///
    /// ```no_run
    /// use std::io;
    /// use std::net::TcpStream;
    /// use std::time::Duration;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080").unwrap();
    /// let result = stream.set_read_timeout(Some(Duration::new(0, 0)));
    /// let err = result.unwrap_err();
    /// assert_eq!(err.kind(), io::ErrorKind::InvalidInput)
    /// ```
    #[stable(feature = "socket_timeout", since = "1.4.0")]
    pub fn set_read_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        self.0.set_read_timeout(dur)
    }

    /// Sets the write timeout to the timeout specified.
    ///
    /// If the value specified is [`None`], then [`write`] calls will block
    /// indefinitely. An [`Err`] is returned if the zero [`Duration`] is
    /// passed to this method.
    ///
    /// # Platform-specific behavior
    ///
    /// Platforms may return a different error code whenever a write times out
    /// as a result of setting this option. For example Unix typically returns
    /// an error of the kind [`WouldBlock`], but Windows may return [`TimedOut`].
    ///
    /// [`write`]: Write::write
    /// [`WouldBlock`]: io::ErrorKind::WouldBlock
    /// [`TimedOut`]: io::ErrorKind::TimedOut
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::TcpStream;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///                        .expect("Couldn't connect to the server...");
    /// stream.set_write_timeout(None).expect("set_write_timeout call failed");
    /// ```
    ///
    /// An [`Err`] is returned if the zero [`Duration`] is passed to this
    /// method:
    ///
    /// ```no_run
    /// use std::io;
    /// use std::net::TcpStream;
    /// use std::time::Duration;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080").unwrap();
    /// let result = stream.set_write_timeout(Some(Duration::new(0, 0)));
    /// let err = result.unwrap_err();
    /// assert_eq!(err.kind(), io::ErrorKind::InvalidInput)
    /// ```
    #[stable(feature = "socket_timeout", since = "1.4.0")]
    pub fn set_write_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        self.0.set_write_timeout(dur)
    }

    /// Returns the read timeout of this socket.
    ///
    /// If the timeout is [`None`], then [`read`] calls will block indefinitely.
    ///
    /// # Platform-specific behavior
    ///
    /// Some platforms do not provide access to the current timeout.
    ///
    /// [`read`]: Read::read
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::TcpStream;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///                        .expect("Couldn't connect to the server...");
    /// stream.set_read_timeout(None).expect("set_read_timeout call failed");
    /// assert_eq!(stream.read_timeout().unwrap(), None);
    /// ```
    #[stable(feature = "socket_timeout", since = "1.4.0")]
    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        self.0.read_timeout()
    }

    /// Returns the write timeout of this socket.
    ///
    /// If the timeout is [`None`], then [`write`] calls will block indefinitely.
    ///
    /// # Platform-specific behavior
    ///
    /// Some platforms do not provide access to the current timeout.
    ///
    /// [`write`]: Write::write
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::TcpStream;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///                        .expect("Couldn't connect to the server...");
    /// stream.set_write_timeout(None).expect("set_write_timeout call failed");
    /// assert_eq!(stream.write_timeout().unwrap(), None);
    /// ```
    #[stable(feature = "socket_timeout", since = "1.4.0")]
    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        self.0.write_timeout()
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
    /// use std::net::TcpStream;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8000")
    ///                        .expect("couldn't bind to address");
    /// let mut buf = [0; 10];
    /// let len = stream.peek(&mut buf).expect("peek failed");
    /// ```
    #[stable(feature = "peek", since = "1.18.0")]
    pub fn peek(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.peek(buf)
    }

    /// Sets the value of the `SO_LINGER` option on this socket.
    ///
    /// This value controls how the socket is closed when data remains
    /// to be sent. If `SO_LINGER` is set, the socket will remain open
    /// for the specified duration as the system attempts to send pending data.
    /// Otherwise, the system may close the socket immediately, or wait for a
    /// default timeout.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(tcp_linger)]
    ///
    /// use std::net::TcpStream;
    /// use std::time::Duration;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///                        .expect("Couldn't connect to the server...");
    /// stream.set_linger(Some(Duration::from_secs(0))).expect("set_linger call failed");
    /// ```
    #[unstable(feature = "tcp_linger", issue = "88494")]
    pub fn set_linger(&self, linger: Option<Duration>) -> io::Result<()> {
        self.0.set_linger(linger)
    }

    /// Gets the value of the `SO_LINGER` option on this socket.
    ///
    /// For more information about this option, see [`TcpStream::set_linger`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(tcp_linger)]
    ///
    /// use std::net::TcpStream;
    /// use std::time::Duration;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///                        .expect("Couldn't connect to the server...");
    /// stream.set_linger(Some(Duration::from_secs(0))).expect("set_linger call failed");
    /// assert_eq!(stream.linger().unwrap(), Some(Duration::from_secs(0)));
    /// ```
    #[unstable(feature = "tcp_linger", issue = "88494")]
    pub fn linger(&self) -> io::Result<Option<Duration>> {
        self.0.linger()
    }

    /// Sets the value of the `TCP_NODELAY` option on this socket.
    ///
    /// If set, this option disables the Nagle algorithm. This means that
    /// segments are always sent as soon as possible, even if there is only a
    /// small amount of data. When not set, data is buffered until there is a
    /// sufficient amount to send out, thereby avoiding the frequent sending of
    /// small packets.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::TcpStream;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///                        .expect("Couldn't connect to the server...");
    /// stream.set_nodelay(true).expect("set_nodelay call failed");
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn set_nodelay(&self, nodelay: bool) -> io::Result<()> {
        self.0.set_nodelay(nodelay)
    }

    /// Gets the value of the `TCP_NODELAY` option on this socket.
    ///
    /// For more information about this option, see [`TcpStream::set_nodelay`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::TcpStream;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///                        .expect("Couldn't connect to the server...");
    /// stream.set_nodelay(true).expect("set_nodelay call failed");
    /// assert_eq!(stream.nodelay().unwrap_or(false), true);
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn nodelay(&self) -> io::Result<bool> {
        self.0.nodelay()
    }

    /// Sets the value for the `IP_TTL` option on this socket.
    ///
    /// This value sets the time-to-live field that is used in every packet sent
    /// from this socket.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::TcpStream;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///                        .expect("Couldn't connect to the server...");
    /// stream.set_ttl(100).expect("set_ttl call failed");
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        self.0.set_ttl(ttl)
    }

    /// Gets the value of the `IP_TTL` option for this socket.
    ///
    /// For more information about this option, see [`TcpStream::set_ttl`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::TcpStream;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///                        .expect("Couldn't connect to the server...");
    /// stream.set_ttl(100).expect("set_ttl call failed");
    /// assert_eq!(stream.ttl().unwrap_or(0), 100);
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn ttl(&self) -> io::Result<u32> {
        self.0.ttl()
    }

    /// Gets the value of the `SO_ERROR` option on this socket.
    ///
    /// This will retrieve the stored error in the underlying socket, clearing
    /// the field in the process. This can be useful for checking errors between
    /// calls.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::TcpStream;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///                        .expect("Couldn't connect to the server...");
    /// stream.take_error().expect("No error was expected...");
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        self.0.take_error()
    }

    /// Moves this TCP stream into or out of nonblocking mode.
    ///
    /// This will result in `read`, `write`, `recv` and `send` operations
    /// becoming nonblocking, i.e., immediately returning from their calls.
    /// If the IO operation is successful, `Ok` is returned and no further
    /// action is required. If the IO operation could not be completed and needs
    /// to be retried, an error with kind [`io::ErrorKind::WouldBlock`] is
    /// returned.
    ///
    /// On Unix platforms, calling this method corresponds to calling `fcntl`
    /// `FIONBIO`. On Windows calling this method corresponds to calling
    /// `ioctlsocket` `FIONBIO`.
    ///
    /// # Examples
    ///
    /// Reading bytes from a TCP stream in non-blocking mode:
    ///
    /// ```no_run
    /// use std::io::{self, Read};
    /// use std::net::TcpStream;
    ///
    /// let mut stream = TcpStream::connect("127.0.0.1:7878")
    ///     .expect("Couldn't connect to the server...");
    /// stream.set_nonblocking(true).expect("set_nonblocking call failed");
    ///
    /// # fn wait_for_fd() { unimplemented!() }
    /// let mut buf = vec![];
    /// loop {
    ///     match stream.read_to_end(&mut buf) {
    ///         Ok(_) => break,
    ///         Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
    ///             // wait until network socket is ready, typically implemented
    ///             // via platform-specific APIs such as epoll or IOCP
    ///             wait_for_fd();
    ///         }
    ///         Err(e) => panic!("encountered IO error: {}", e),
    ///     };
    /// };
    /// println!("bytes: {:?}", buf);
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.0.set_nonblocking(nonblocking)
    }
}

// In addition to the `impl`s here, `TcpStream` also has `impl`s for
// `AsFd`/`From<OwnedFd>`/`Into<OwnedFd>` and
// `AsRawFd`/`IntoRawFd`/`FromRawFd`, on Unix and WASI, and
// `AsSocket`/`From<OwnedSocket>`/`Into<OwnedSocket>` and
// `AsRawSocket`/`IntoRawSocket`/`FromRawSocket` on Windows.

#[stable(feature = "rust1", since = "1.0.0")]
impl Read for TcpStream {
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
#[stable(feature = "rust1", since = "1.0.0")]
impl Write for TcpStream {
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
#[stable(feature = "rust1", since = "1.0.0")]
impl Read for &TcpStream {
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
#[stable(feature = "rust1", since = "1.0.0")]
impl Write for &TcpStream {
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

impl AsInner<net_imp::TcpStream> for TcpStream {
    fn as_inner(&self) -> &net_imp::TcpStream {
        &self.0
    }
}

impl FromInner<net_imp::TcpStream> for TcpStream {
    fn from_inner(inner: net_imp::TcpStream) -> TcpStream {
        TcpStream(inner)
    }
}

impl IntoInner<net_imp::TcpStream> for TcpStream {
    fn into_inner(self) -> net_imp::TcpStream {
        self.0
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for TcpStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl TcpListener {
    /// Creates a new `TcpListener` which will be bound to the specified
    /// address.
    ///
    /// The returned listener is ready for accepting connections.
    ///
    /// Binding with a port number of 0 will request that the OS assigns a port
    /// to this listener. The port allocated can be queried via the
    /// [`TcpListener::local_addr`] method.
    ///
    /// The address type can be any implementor of [`ToSocketAddrs`] trait. See
    /// its documentation for concrete examples.
    ///
    /// If `addr` yields multiple addresses, `bind` will be attempted with
    /// each of the addresses until one succeeds and returns the listener. If
    /// none of the addresses succeed in creating a listener, the error returned
    /// from the last attempt (the last address) is returned.
    ///
    /// # Examples
    ///
    /// Creates a TCP listener bound to `127.0.0.1:80`:
    ///
    /// ```no_run
    /// use std::net::TcpListener;
    ///
    /// let listener = TcpListener::bind("127.0.0.1:80").unwrap();
    /// ```
    ///
    /// Creates a TCP listener bound to `127.0.0.1:80`. If that fails, create a
    /// TCP listener bound to `127.0.0.1:443`:
    ///
    /// ```no_run
    /// use std::net::{SocketAddr, TcpListener};
    ///
    /// let addrs = [
    ///     SocketAddr::from(([127, 0, 0, 1], 80)),
    ///     SocketAddr::from(([127, 0, 0, 1], 443)),
    /// ];
    /// let listener = TcpListener::bind(&addrs[..]).unwrap();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn bind<A: ToSocketAddrs>(addr: A) -> io::Result<TcpListener> {
        super::each_addr(addr, net_imp::TcpListener::bind).map(TcpListener)
    }

    /// Returns the local socket address of this listener.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener};
    ///
    /// let listener = TcpListener::bind("127.0.0.1:8080").unwrap();
    /// assert_eq!(listener.local_addr().unwrap(),
    ///            SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 8080)));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.0.socket_addr()
    }

    /// Creates a new independently owned handle to the underlying socket.
    ///
    /// The returned [`TcpListener`] is a reference to the same socket that this
    /// object references. Both handles can be used to accept incoming
    /// connections and options set on one listener will affect the other.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::TcpListener;
    ///
    /// let listener = TcpListener::bind("127.0.0.1:8080").unwrap();
    /// let listener_clone = listener.try_clone().unwrap();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn try_clone(&self) -> io::Result<TcpListener> {
        self.0.duplicate().map(TcpListener)
    }

    /// Accept a new incoming connection from this listener.
    ///
    /// This function will block the calling thread until a new TCP connection
    /// is established. When established, the corresponding [`TcpStream`] and the
    /// remote peer's address will be returned.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::TcpListener;
    ///
    /// let listener = TcpListener::bind("127.0.0.1:8080").unwrap();
    /// match listener.accept() {
    ///     Ok((_socket, addr)) => println!("new client: {:?}", addr),
    ///     Err(e) => println!("couldn't get client: {:?}", e),
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        // On WASM, `TcpStream` is uninhabited (as it's unsupported) and so
        // the `a` variable here is technically unused.
        #[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
        self.0.accept().map(|(a, b)| (TcpStream(a), b))
    }

    /// Returns an iterator over the connections being received on this
    /// listener.
    ///
    /// The returned iterator will never return [`None`] and will also not yield
    /// the peer's [`SocketAddr`] structure. Iterating over it is equivalent to
    /// calling [`TcpListener::accept`] in a loop.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::{TcpListener, TcpStream};
    ///
    /// fn handle_connection(stream: TcpStream) {
    ///    //...
    /// }
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let listener = TcpListener::bind("127.0.0.1:80").unwrap();
    ///
    ///     for stream in listener.incoming() {
    ///         match stream {
    ///             Ok(stream) => {
    ///                 handle_connection(stream);
    ///             }
    ///             Err(e) => { /* connection failed */ }
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn incoming(&self) -> Incoming<'_> {
        Incoming { listener: self }
    }

    /// Turn this into an iterator over the connections being received on this
    /// listener.
    ///
    /// The returned iterator will never return [`None`] and will also not yield
    /// the peer's [`SocketAddr`] structure. Iterating over it is equivalent to
    /// calling [`TcpListener::accept`] in a loop.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(tcplistener_into_incoming)]
    /// use std::net::{TcpListener, TcpStream};
    ///
    /// fn listen_on(port: u16) -> impl Iterator<Item = TcpStream> {
    ///     let listener = TcpListener::bind("127.0.0.1:80").unwrap();
    ///     listener.into_incoming()
    ///         .filter_map(Result::ok) /* Ignore failed connections */
    /// }
    ///
    /// fn main() -> std::io::Result<()> {
    ///     for stream in listen_on(80) {
    ///         /* handle the connection here */
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]
    #[unstable(feature = "tcplistener_into_incoming", issue = "88339")]
    pub fn into_incoming(self) -> IntoIncoming {
        IntoIncoming { listener: self }
    }

    /// Sets the value for the `IP_TTL` option on this socket.
    ///
    /// This value sets the time-to-live field that is used in every packet sent
    /// from this socket.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::TcpListener;
    ///
    /// let listener = TcpListener::bind("127.0.0.1:80").unwrap();
    /// listener.set_ttl(100).expect("could not set TTL");
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        self.0.set_ttl(ttl)
    }

    /// Gets the value of the `IP_TTL` option for this socket.
    ///
    /// For more information about this option, see [`TcpListener::set_ttl`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::TcpListener;
    ///
    /// let listener = TcpListener::bind("127.0.0.1:80").unwrap();
    /// listener.set_ttl(100).expect("could not set TTL");
    /// assert_eq!(listener.ttl().unwrap_or(0), 100);
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn ttl(&self) -> io::Result<u32> {
        self.0.ttl()
    }

    #[stable(feature = "net2_mutators", since = "1.9.0")]
    #[rustc_deprecated(
        since = "1.16.0",
        reason = "this option can only be set before the socket is bound"
    )]
    #[allow(missing_docs)]
    pub fn set_only_v6(&self, only_v6: bool) -> io::Result<()> {
        self.0.set_only_v6(only_v6)
    }

    #[stable(feature = "net2_mutators", since = "1.9.0")]
    #[rustc_deprecated(
        since = "1.16.0",
        reason = "this option can only be set before the socket is bound"
    )]
    #[allow(missing_docs)]
    pub fn only_v6(&self) -> io::Result<bool> {
        self.0.only_v6()
    }

    /// Gets the value of the `SO_ERROR` option on this socket.
    ///
    /// This will retrieve the stored error in the underlying socket, clearing
    /// the field in the process. This can be useful for checking errors between
    /// calls.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::TcpListener;
    ///
    /// let listener = TcpListener::bind("127.0.0.1:80").unwrap();
    /// listener.take_error().expect("No error was expected");
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        self.0.take_error()
    }

    /// Moves this TCP stream into or out of nonblocking mode.
    ///
    /// This will result in the `accept` operation becoming nonblocking,
    /// i.e., immediately returning from their calls. If the IO operation is
    /// successful, `Ok` is returned and no further action is required. If the
    /// IO operation could not be completed and needs to be retried, an error
    /// with kind [`io::ErrorKind::WouldBlock`] is returned.
    ///
    /// On Unix platforms, calling this method corresponds to calling `fcntl`
    /// `FIONBIO`. On Windows calling this method corresponds to calling
    /// `ioctlsocket` `FIONBIO`.
    ///
    /// # Examples
    ///
    /// Bind a TCP listener to an address, listen for connections, and read
    /// bytes in nonblocking mode:
    ///
    /// ```no_run
    /// use std::io;
    /// use std::net::TcpListener;
    ///
    /// let listener = TcpListener::bind("127.0.0.1:7878").unwrap();
    /// listener.set_nonblocking(true).expect("Cannot set non-blocking");
    ///
    /// # fn wait_for_fd() { unimplemented!() }
    /// # fn handle_connection(stream: std::net::TcpStream) { unimplemented!() }
    /// for stream in listener.incoming() {
    ///     match stream {
    ///         Ok(s) => {
    ///             // do something with the TcpStream
    ///             handle_connection(s);
    ///         }
    ///         Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
    ///             // wait until network socket is ready, typically implemented
    ///             // via platform-specific APIs such as epoll or IOCP
    ///             wait_for_fd();
    ///             continue;
    ///         }
    ///         Err(e) => panic!("encountered IO error: {}", e),
    ///     }
    /// }
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.0.set_nonblocking(nonblocking)
    }
}

// In addition to the `impl`s here, `TcpListener` also has `impl`s for
// `AsFd`/`From<OwnedFd>`/`Into<OwnedFd>` and
// `AsRawFd`/`IntoRawFd`/`FromRawFd`, on Unix and WASI, and
// `AsSocket`/`From<OwnedSocket>`/`Into<OwnedSocket>` and
// `AsRawSocket`/`IntoRawSocket`/`FromRawSocket` on Windows.

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Incoming<'a> {
    type Item = io::Result<TcpStream>;
    fn next(&mut self) -> Option<io::Result<TcpStream>> {
        Some(self.listener.accept().map(|p| p.0))
    }
}

#[unstable(feature = "tcplistener_into_incoming", issue = "88339")]
impl Iterator for IntoIncoming {
    type Item = io::Result<TcpStream>;
    fn next(&mut self) -> Option<io::Result<TcpStream>> {
        Some(self.listener.accept().map(|p| p.0))
    }
}

impl AsInner<net_imp::TcpListener> for TcpListener {
    fn as_inner(&self) -> &net_imp::TcpListener {
        &self.0
    }
}

impl FromInner<net_imp::TcpListener> for TcpListener {
    fn from_inner(inner: net_imp::TcpListener) -> TcpListener {
        TcpListener(inner)
    }
}

impl IntoInner<net_imp::TcpListener> for TcpListener {
    fn into_inner(self) -> net_imp::TcpListener {
        self.0
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for TcpListener {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
