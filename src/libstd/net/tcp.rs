// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io::prelude::*;

use fmt;
use io;
use net::{ToSocketAddrs, SocketAddr, Shutdown};
use sys_common::net as net_imp;
use sys_common::{AsInner, FromInner, IntoInner};
use time::Duration;

/// A structure which represents a TCP stream between a local socket and a
/// remote socket.
///
/// The socket will be closed when the value is dropped.
///
/// # Examples
///
/// ```no_run
/// use std::io::prelude::*;
/// use std::net::TcpStream;
///
/// {
///     let mut stream = TcpStream::connect("127.0.0.1:34254").unwrap();
///
///     // ignore the Result
///     let _ = stream.write(&[1]);
///     let _ = stream.read(&mut [0; 128]); // ignore here too
/// } // the stream is closed here
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct TcpStream(net_imp::TcpStream);

/// A structure representing a socket server.
///
/// # Examples
///
/// ```no_run
/// use std::net::{TcpListener, TcpStream};
///
/// let listener = TcpListener::bind("127.0.0.1:80").unwrap();
///
/// fn handle_client(stream: TcpStream) {
///     // ...
/// }
///
/// // accept connections and process them serially
/// for stream in listener.incoming() {
///     match stream {
///         Ok(stream) => {
///             handle_client(stream);
///         }
///         Err(e) => { /* connection failed */ }
///     }
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct TcpListener(net_imp::TcpListener);

/// An infinite iterator over the connections from a `TcpListener`.
///
/// This iterator will infinitely yield [`Some`] of the accepted connections. It
/// is equivalent to calling `accept` in a loop.
///
/// This `struct` is created by the [`incoming`] method on [`TcpListener`].
///
/// [`Some`]: ../../std/option/enum.Option.html#variant.Some
/// [`incoming`]: struct.TcpListener.html#method.incoming
/// [`TcpListener`]: struct.TcpListener.html
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug)]
pub struct Incoming<'a> { listener: &'a TcpListener }

impl TcpStream {
    /// Opens a TCP connection to a remote host.
    ///
    /// `addr` is an address of the remote host. Anything which implements
    /// `ToSocketAddrs` trait can be supplied for the address; see this trait
    /// documentation for concrete examples.
    /// In case `ToSocketAddrs::to_socket_addrs()` returns more than one entry,
    /// then the first valid and reachable address is used.
    ///
    /// # Examples
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn connect<A: ToSocketAddrs>(addr: A) -> io::Result<TcpStream> {
        super::each_addr(addr, net_imp::TcpStream::connect).map(TcpStream)
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
    /// use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream};
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///                        .expect("Couldn't connect to the server...");
    /// assert_eq!(stream.local_addr().unwrap(),
    ///            SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 8080)));
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
    /// [`Shutdown`]: ../../std/net/enum.Shutdown.html
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
    /// If the value specified is [`None`], then [`read()`] calls will block
    /// indefinitely. It is an error to pass the zero `Duration` to this
    /// method.
    ///
    /// # Note
    ///
    /// Platforms may return a different error code whenever a read times out as
    /// a result of setting this option. For example Unix typically returns an
    /// error of the kind [`WouldBlock`], but Windows may return [`TimedOut`].
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    /// [`read()`]: ../../std/io/trait.Read.html#tymethod.read
    /// [`WouldBlock`]: ../../std/io/enum.ErrorKind.html#variant.WouldBlock
    /// [`TimedOut`]: ../../std/io/enum.ErrorKind.html#variant.TimedOut
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
    #[stable(feature = "socket_timeout", since = "1.4.0")]
    pub fn set_read_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        self.0.set_read_timeout(dur)
    }

    /// Sets the write timeout to the timeout specified.
    ///
    /// If the value specified is [`None`], then [`write()`] calls will block
    /// indefinitely. It is an error to pass the zero [`Duration`] to this
    /// method.
    ///
    /// # Note
    ///
    /// Platforms may return a different error code whenever a write times out
    /// as a result of setting this option. For example Unix typically returns
    /// an error of the kind [`WouldBlock`], but Windows may return [`TimedOut`].
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    /// [`write()`]: ../../std/io/trait.Write.html#tymethod.write
    /// [`Duration`]: ../../std/time/struct.Duration.html
    /// [`WouldBlock`]: ../../std/io/enum.ErrorKind.html#variant.WouldBlock
    /// [`TimedOut`]: ../../std/io/enum.ErrorKind.html#variant.TimedOut
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
    #[stable(feature = "socket_timeout", since = "1.4.0")]
    pub fn set_write_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        self.0.set_write_timeout(dur)
    }

    /// Returns the read timeout of this socket.
    ///
    /// If the timeout is [`None`], then [`read()`] calls will block indefinitely.
    ///
    /// # Note
    ///
    /// Some platforms do not provide access to the current timeout.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    /// [`read()`]: ../../std/io/trait.Read.html#tymethod.read
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
    /// If the timeout is [`None`], then [`write()`] calls will block indefinitely.
    ///
    /// # Note
    ///
    /// Some platforms do not provide access to the current timeout.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    /// [`write()`]: ../../std/io/trait.Write.html#tymethod.write
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
    /// For more information about this option, see [`set_nodelay`][link].
    ///
    /// [link]: #method.set_nodelay
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
    /// For more information about this option, see [`set_ttl`][link].
    ///
    /// [link]: #method.set_ttl
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

    /// Get the value of the `SO_ERROR` option on this socket.
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
    /// On Unix this corresponds to calling fcntl, and on Windows this
    /// corresponds to calling ioctlsocket.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::TcpStream;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///                        .expect("Couldn't connect to the server...");
    /// stream.set_nonblocking(true).expect("set_nonblocking call failed");
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.0.set_nonblocking(nonblocking)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Read for TcpStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> { self.0.read(buf) }
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        self.0.read_to_end(buf)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl Write for TcpStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> { self.0.write(buf) }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Read for &'a TcpStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> { self.0.read(buf) }
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        self.0.read_to_end(buf)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Write for &'a TcpStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> { self.0.write(buf) }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

impl AsInner<net_imp::TcpStream> for TcpStream {
    fn as_inner(&self) -> &net_imp::TcpStream { &self.0 }
}

impl FromInner<net_imp::TcpStream> for TcpStream {
    fn from_inner(inner: net_imp::TcpStream) -> TcpStream { TcpStream(inner) }
}

impl IntoInner<net_imp::TcpStream> for TcpStream {
    fn into_inner(self) -> net_imp::TcpStream { self.0 }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for TcpStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
    /// `local_addr` method.
    ///
    /// The address type can be any implementor of `ToSocketAddrs` trait. See
    /// its documentation for concrete examples.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::TcpListener;
    ///
    /// let listener = TcpListener::bind("127.0.0.1:80").unwrap();
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
    /// The returned `TcpListener` is a reference to the same socket that this
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
    /// is established. When established, the corresponding `TcpStream` and the
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
        self.0.accept().map(|(a, b)| (TcpStream(a), b))
    }

    /// Returns an iterator over the connections being received on this
    /// listener.
    ///
    /// The returned iterator will never return [`None`] and will also not yield
    /// the peer's [`SocketAddr`] structure.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    /// [`SocketAddr`]: ../../std/net/enum.SocketAddr.html
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::TcpListener;
    ///
    /// let listener = TcpListener::bind("127.0.0.1:80").unwrap();
    ///
    /// for stream in listener.incoming() {
    ///     match stream {
    ///         Ok(stream) => {
    ///             println!("new client!");
    ///         }
    ///         Err(e) => { /* connection failed */ }
    ///     }
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn incoming(&self) -> Incoming {
        Incoming { listener: self }
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
    /// For more information about this option, see [`set_ttl()`][link].
    ///
    /// [link]: #method.set_ttl
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
    #[rustc_deprecated(since = "1.16.0",
                       reason = "this option can only be set before the socket is bound")]
    #[allow(missing_docs)]
    pub fn set_only_v6(&self, only_v6: bool) -> io::Result<()> {
        self.0.set_only_v6(only_v6)
    }

    #[stable(feature = "net2_mutators", since = "1.9.0")]
    #[rustc_deprecated(since = "1.16.0",
                       reason = "this option can only be set before the socket is bound")]
    #[allow(missing_docs)]
    pub fn only_v6(&self) -> io::Result<bool> {
        self.0.only_v6()
    }

    /// Get the value of the `SO_ERROR` option on this socket.
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
    /// On Unix this corresponds to calling fcntl, and on Windows this
    /// corresponds to calling ioctlsocket.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::TcpListener;
    ///
    /// let listener = TcpListener::bind("127.0.0.1:80").unwrap();
    /// listener.set_nonblocking(true).expect("Cannot set non-blocking");
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.0.set_nonblocking(nonblocking)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Incoming<'a> {
    type Item = io::Result<TcpStream>;
    fn next(&mut self) -> Option<io::Result<TcpStream>> {
        Some(self.listener.accept().map(|p| p.0))
    }
}

impl AsInner<net_imp::TcpListener> for TcpListener {
    fn as_inner(&self) -> &net_imp::TcpListener { &self.0 }
}

impl FromInner<net_imp::TcpListener> for TcpListener {
    fn from_inner(inner: net_imp::TcpListener) -> TcpListener {
        TcpListener(inner)
    }
}

impl IntoInner<net_imp::TcpListener> for TcpListener {
    fn into_inner(self) -> net_imp::TcpListener { self.0 }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for TcpListener {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(all(test, not(target_os = "emscripten")))]
mod tests {
    use io::ErrorKind;
    use io::prelude::*;
    use net::*;
    use net::test::{next_test_ip4, next_test_ip6};
    use sync::mpsc::channel;
    use sys_common::AsInner;
    use time::{Instant, Duration};
    use thread;

    fn each_ip(f: &mut FnMut(SocketAddr)) {
        f(next_test_ip4());
        f(next_test_ip6());
    }

    macro_rules! t {
        ($e:expr) => {
            match $e {
                Ok(t) => t,
                Err(e) => panic!("received error for `{}`: {}", stringify!($e), e),
            }
        }
    }

    #[test]
    fn bind_error() {
        match TcpListener::bind("1.1.1.1:9999") {
            Ok(..) => panic!(),
            Err(e) =>
                assert_eq!(e.kind(), ErrorKind::AddrNotAvailable),
        }
    }

    #[test]
    fn connect_error() {
        match TcpStream::connect("0.0.0.0:1") {
            Ok(..) => panic!(),
            Err(e) => assert!(e.kind() == ErrorKind::ConnectionRefused ||
                              e.kind() == ErrorKind::InvalidInput ||
                              e.kind() == ErrorKind::AddrInUse ||
                              e.kind() == ErrorKind::AddrNotAvailable,
                              "bad error: {} {:?}", e, e.kind()),
        }
    }

    #[test]
    fn listen_localhost() {
        let socket_addr = next_test_ip4();
        let listener = t!(TcpListener::bind(&socket_addr));

        let _t = thread::spawn(move || {
            let mut stream = t!(TcpStream::connect(&("localhost",
                                                     socket_addr.port())));
            t!(stream.write(&[144]));
        });

        let mut stream = t!(listener.accept()).0;
        let mut buf = [0];
        t!(stream.read(&mut buf));
        assert!(buf[0] == 144);
    }

    #[test]
    fn connect_loopback() {
        each_ip(&mut |addr| {
            let acceptor = t!(TcpListener::bind(&addr));

            let _t = thread::spawn(move|| {
                let host = match addr {
                    SocketAddr::V4(..) => "127.0.0.1",
                    SocketAddr::V6(..) => "::1",
                };
                let mut stream = t!(TcpStream::connect(&(host, addr.port())));
                t!(stream.write(&[66]));
            });

            let mut stream = t!(acceptor.accept()).0;
            let mut buf = [0];
            t!(stream.read(&mut buf));
            assert!(buf[0] == 66);
        })
    }

    #[test]
    fn smoke_test() {
        each_ip(&mut |addr| {
            let acceptor = t!(TcpListener::bind(&addr));

            let (tx, rx) = channel();
            let _t = thread::spawn(move|| {
                let mut stream = t!(TcpStream::connect(&addr));
                t!(stream.write(&[99]));
                tx.send(t!(stream.local_addr())).unwrap();
            });

            let (mut stream, addr) = t!(acceptor.accept());
            let mut buf = [0];
            t!(stream.read(&mut buf));
            assert!(buf[0] == 99);
            assert_eq!(addr, t!(rx.recv()));
        })
    }

    #[test]
    fn read_eof() {
        each_ip(&mut |addr| {
            let acceptor = t!(TcpListener::bind(&addr));

            let _t = thread::spawn(move|| {
                let _stream = t!(TcpStream::connect(&addr));
                // Close
            });

            let mut stream = t!(acceptor.accept()).0;
            let mut buf = [0];
            let nread = t!(stream.read(&mut buf));
            assert_eq!(nread, 0);
            let nread = t!(stream.read(&mut buf));
            assert_eq!(nread, 0);
        })
    }

    #[test]
    fn write_close() {
        each_ip(&mut |addr| {
            let acceptor = t!(TcpListener::bind(&addr));

            let (tx, rx) = channel();
            let _t = thread::spawn(move|| {
                drop(t!(TcpStream::connect(&addr)));
                tx.send(()).unwrap();
            });

            let mut stream = t!(acceptor.accept()).0;
            rx.recv().unwrap();
            let buf = [0];
            match stream.write(&buf) {
                Ok(..) => {}
                Err(e) => {
                    assert!(e.kind() == ErrorKind::ConnectionReset ||
                            e.kind() == ErrorKind::BrokenPipe ||
                            e.kind() == ErrorKind::ConnectionAborted,
                            "unknown error: {}", e);
                }
            }
        })
    }

    #[test]
    fn multiple_connect_serial() {
        each_ip(&mut |addr| {
            let max = 10;
            let acceptor = t!(TcpListener::bind(&addr));

            let _t = thread::spawn(move|| {
                for _ in 0..max {
                    let mut stream = t!(TcpStream::connect(&addr));
                    t!(stream.write(&[99]));
                }
            });

            for stream in acceptor.incoming().take(max) {
                let mut stream = t!(stream);
                let mut buf = [0];
                t!(stream.read(&mut buf));
                assert_eq!(buf[0], 99);
            }
        })
    }

    #[test]
    fn multiple_connect_interleaved_greedy_schedule() {
        const MAX: usize = 10;
        each_ip(&mut |addr| {
            let acceptor = t!(TcpListener::bind(&addr));

            let _t = thread::spawn(move|| {
                let acceptor = acceptor;
                for (i, stream) in acceptor.incoming().enumerate().take(MAX) {
                    // Start another thread to handle the connection
                    let _t = thread::spawn(move|| {
                        let mut stream = t!(stream);
                        let mut buf = [0];
                        t!(stream.read(&mut buf));
                        assert!(buf[0] == i as u8);
                    });
                }
            });

            connect(0, addr);
        });

        fn connect(i: usize, addr: SocketAddr) {
            if i == MAX { return }

            let t = thread::spawn(move|| {
                let mut stream = t!(TcpStream::connect(&addr));
                // Connect again before writing
                connect(i + 1, addr);
                t!(stream.write(&[i as u8]));
            });
            t.join().ok().unwrap();
        }
    }

    #[test]
    fn multiple_connect_interleaved_lazy_schedule() {
        const MAX: usize = 10;
        each_ip(&mut |addr| {
            let acceptor = t!(TcpListener::bind(&addr));

            let _t = thread::spawn(move|| {
                for stream in acceptor.incoming().take(MAX) {
                    // Start another thread to handle the connection
                    let _t = thread::spawn(move|| {
                        let mut stream = t!(stream);
                        let mut buf = [0];
                        t!(stream.read(&mut buf));
                        assert!(buf[0] == 99);
                    });
                }
            });

            connect(0, addr);
        });

        fn connect(i: usize, addr: SocketAddr) {
            if i == MAX { return }

            let t = thread::spawn(move|| {
                let mut stream = t!(TcpStream::connect(&addr));
                connect(i + 1, addr);
                t!(stream.write(&[99]));
            });
            t.join().ok().unwrap();
        }
    }

    #[test]
    fn socket_and_peer_name() {
        each_ip(&mut |addr| {
            let listener = t!(TcpListener::bind(&addr));
            let so_name = t!(listener.local_addr());
            assert_eq!(addr, so_name);
            let _t = thread::spawn(move|| {
                t!(listener.accept());
            });

            let stream = t!(TcpStream::connect(&addr));
            assert_eq!(addr, t!(stream.peer_addr()));
        })
    }

    #[test]
    fn partial_read() {
        each_ip(&mut |addr| {
            let (tx, rx) = channel();
            let srv = t!(TcpListener::bind(&addr));
            let _t = thread::spawn(move|| {
                let mut cl = t!(srv.accept()).0;
                cl.write(&[10]).unwrap();
                let mut b = [0];
                t!(cl.read(&mut b));
                tx.send(()).unwrap();
            });

            let mut c = t!(TcpStream::connect(&addr));
            let mut b = [0; 10];
            assert_eq!(c.read(&mut b).unwrap(), 1);
            t!(c.write(&[1]));
            rx.recv().unwrap();
        })
    }

    #[test]
    fn double_bind() {
        each_ip(&mut |addr| {
            let _listener = t!(TcpListener::bind(&addr));
            match TcpListener::bind(&addr) {
                Ok(..) => panic!(),
                Err(e) => {
                    assert!(e.kind() == ErrorKind::ConnectionRefused ||
                            e.kind() == ErrorKind::Other ||
                            e.kind() == ErrorKind::AddrInUse,
                            "unknown error: {} {:?}", e, e.kind());
                }
            }
        })
    }

    #[test]
    fn fast_rebind() {
        each_ip(&mut |addr| {
            let acceptor = t!(TcpListener::bind(&addr));

            let _t = thread::spawn(move|| {
                t!(TcpStream::connect(&addr));
            });

            t!(acceptor.accept());
            drop(acceptor);
            t!(TcpListener::bind(&addr));
        });
    }

    #[test]
    fn tcp_clone_smoke() {
        each_ip(&mut |addr| {
            let acceptor = t!(TcpListener::bind(&addr));

            let _t = thread::spawn(move|| {
                let mut s = t!(TcpStream::connect(&addr));
                let mut buf = [0, 0];
                assert_eq!(s.read(&mut buf).unwrap(), 1);
                assert_eq!(buf[0], 1);
                t!(s.write(&[2]));
            });

            let mut s1 = t!(acceptor.accept()).0;
            let s2 = t!(s1.try_clone());

            let (tx1, rx1) = channel();
            let (tx2, rx2) = channel();
            let _t = thread::spawn(move|| {
                let mut s2 = s2;
                rx1.recv().unwrap();
                t!(s2.write(&[1]));
                tx2.send(()).unwrap();
            });
            tx1.send(()).unwrap();
            let mut buf = [0, 0];
            assert_eq!(s1.read(&mut buf).unwrap(), 1);
            rx2.recv().unwrap();
        })
    }

    #[test]
    fn tcp_clone_two_read() {
        each_ip(&mut |addr| {
            let acceptor = t!(TcpListener::bind(&addr));
            let (tx1, rx) = channel();
            let tx2 = tx1.clone();

            let _t = thread::spawn(move|| {
                let mut s = t!(TcpStream::connect(&addr));
                t!(s.write(&[1]));
                rx.recv().unwrap();
                t!(s.write(&[2]));
                rx.recv().unwrap();
            });

            let mut s1 = t!(acceptor.accept()).0;
            let s2 = t!(s1.try_clone());

            let (done, rx) = channel();
            let _t = thread::spawn(move|| {
                let mut s2 = s2;
                let mut buf = [0, 0];
                t!(s2.read(&mut buf));
                tx2.send(()).unwrap();
                done.send(()).unwrap();
            });
            let mut buf = [0, 0];
            t!(s1.read(&mut buf));
            tx1.send(()).unwrap();

            rx.recv().unwrap();
        })
    }

    #[test]
    fn tcp_clone_two_write() {
        each_ip(&mut |addr| {
            let acceptor = t!(TcpListener::bind(&addr));

            let _t = thread::spawn(move|| {
                let mut s = t!(TcpStream::connect(&addr));
                let mut buf = [0, 1];
                t!(s.read(&mut buf));
                t!(s.read(&mut buf));
            });

            let mut s1 = t!(acceptor.accept()).0;
            let s2 = t!(s1.try_clone());

            let (done, rx) = channel();
            let _t = thread::spawn(move|| {
                let mut s2 = s2;
                t!(s2.write(&[1]));
                done.send(()).unwrap();
            });
            t!(s1.write(&[2]));

            rx.recv().unwrap();
        })
    }

    #[test]
    fn shutdown_smoke() {
        each_ip(&mut |addr| {
            let a = t!(TcpListener::bind(&addr));
            let _t = thread::spawn(move|| {
                let mut c = t!(a.accept()).0;
                let mut b = [0];
                assert_eq!(c.read(&mut b).unwrap(), 0);
                t!(c.write(&[1]));
            });

            let mut s = t!(TcpStream::connect(&addr));
            t!(s.shutdown(Shutdown::Write));
            assert!(s.write(&[1]).is_err());
            let mut b = [0, 0];
            assert_eq!(t!(s.read(&mut b)), 1);
            assert_eq!(b[0], 1);
        })
    }

    #[test]
    fn close_readwrite_smoke() {
        each_ip(&mut |addr| {
            let a = t!(TcpListener::bind(&addr));
            let (tx, rx) = channel::<()>();
            let _t = thread::spawn(move|| {
                let _s = t!(a.accept());
                let _ = rx.recv();
            });

            let mut b = [0];
            let mut s = t!(TcpStream::connect(&addr));
            let mut s2 = t!(s.try_clone());

            // closing should prevent reads/writes
            t!(s.shutdown(Shutdown::Write));
            assert!(s.write(&[0]).is_err());
            t!(s.shutdown(Shutdown::Read));
            assert_eq!(s.read(&mut b).unwrap(), 0);

            // closing should affect previous handles
            assert!(s2.write(&[0]).is_err());
            assert_eq!(s2.read(&mut b).unwrap(), 0);

            // closing should affect new handles
            let mut s3 = t!(s.try_clone());
            assert!(s3.write(&[0]).is_err());
            assert_eq!(s3.read(&mut b).unwrap(), 0);

            // make sure these don't die
            let _ = s2.shutdown(Shutdown::Read);
            let _ = s2.shutdown(Shutdown::Write);
            let _ = s3.shutdown(Shutdown::Read);
            let _ = s3.shutdown(Shutdown::Write);
            drop(tx);
        })
    }

    #[test]
    #[cfg(unix)] // test doesn't work on Windows, see #31657
    fn close_read_wakes_up() {
        each_ip(&mut |addr| {
            let a = t!(TcpListener::bind(&addr));
            let (tx1, rx) = channel::<()>();
            let _t = thread::spawn(move|| {
                let _s = t!(a.accept());
                let _ = rx.recv();
            });

            let s = t!(TcpStream::connect(&addr));
            let s2 = t!(s.try_clone());
            let (tx, rx) = channel();
            let _t = thread::spawn(move|| {
                let mut s2 = s2;
                assert_eq!(t!(s2.read(&mut [0])), 0);
                tx.send(()).unwrap();
            });
            // this should wake up the child thread
            t!(s.shutdown(Shutdown::Read));

            // this test will never finish if the child doesn't wake up
            rx.recv().unwrap();
            drop(tx1);
        })
    }

    #[test]
    fn clone_while_reading() {
        each_ip(&mut |addr| {
            let accept = t!(TcpListener::bind(&addr));

            // Enqueue a thread to write to a socket
            let (tx, rx) = channel();
            let (txdone, rxdone) = channel();
            let txdone2 = txdone.clone();
            let _t = thread::spawn(move|| {
                let mut tcp = t!(TcpStream::connect(&addr));
                rx.recv().unwrap();
                t!(tcp.write(&[0]));
                txdone2.send(()).unwrap();
            });

            // Spawn off a reading clone
            let tcp = t!(accept.accept()).0;
            let tcp2 = t!(tcp.try_clone());
            let txdone3 = txdone.clone();
            let _t = thread::spawn(move|| {
                let mut tcp2 = tcp2;
                t!(tcp2.read(&mut [0]));
                txdone3.send(()).unwrap();
            });

            // Try to ensure that the reading clone is indeed reading
            for _ in 0..50 {
                thread::yield_now();
            }

            // clone the handle again while it's reading, then let it finish the
            // read.
            let _ = t!(tcp.try_clone());
            tx.send(()).unwrap();
            rxdone.recv().unwrap();
            rxdone.recv().unwrap();
        })
    }

    #[test]
    fn clone_accept_smoke() {
        each_ip(&mut |addr| {
            let a = t!(TcpListener::bind(&addr));
            let a2 = t!(a.try_clone());

            let _t = thread::spawn(move|| {
                let _ = TcpStream::connect(&addr);
            });
            let _t = thread::spawn(move|| {
                let _ = TcpStream::connect(&addr);
            });

            t!(a.accept());
            t!(a2.accept());
        })
    }

    #[test]
    fn clone_accept_concurrent() {
        each_ip(&mut |addr| {
            let a = t!(TcpListener::bind(&addr));
            let a2 = t!(a.try_clone());

            let (tx, rx) = channel();
            let tx2 = tx.clone();

            let _t = thread::spawn(move|| {
                tx.send(t!(a.accept())).unwrap();
            });
            let _t = thread::spawn(move|| {
                tx2.send(t!(a2.accept())).unwrap();
            });

            let _t = thread::spawn(move|| {
                let _ = TcpStream::connect(&addr);
            });
            let _t = thread::spawn(move|| {
                let _ = TcpStream::connect(&addr);
            });

            rx.recv().unwrap();
            rx.recv().unwrap();
        })
    }

    #[test]
    fn debug() {
        let name = if cfg!(windows) {"socket"} else {"fd"};
        let socket_addr = next_test_ip4();

        let listener = t!(TcpListener::bind(&socket_addr));
        let listener_inner = listener.0.socket().as_inner();
        let compare = format!("TcpListener {{ addr: {:?}, {}: {:?} }}",
                              socket_addr, name, listener_inner);
        assert_eq!(format!("{:?}", listener), compare);

        let stream = t!(TcpStream::connect(&("localhost",
                                                 socket_addr.port())));
        let stream_inner = stream.0.socket().as_inner();
        let compare = format!("TcpStream {{ addr: {:?}, \
                              peer: {:?}, {}: {:?} }}",
                              stream.local_addr().unwrap(),
                              stream.peer_addr().unwrap(),
                              name,
                              stream_inner);
        assert_eq!(format!("{:?}", stream), compare);
    }

    // FIXME: re-enabled bitrig/openbsd tests once their socket timeout code
    //        no longer has rounding errors.
    #[cfg_attr(any(target_os = "bitrig", target_os = "netbsd", target_os = "openbsd"), ignore)]
    #[test]
    fn timeouts() {
        let addr = next_test_ip4();
        let listener = t!(TcpListener::bind(&addr));

        let stream = t!(TcpStream::connect(&("localhost", addr.port())));
        let dur = Duration::new(15410, 0);

        assert_eq!(None, t!(stream.read_timeout()));

        t!(stream.set_read_timeout(Some(dur)));
        assert_eq!(Some(dur), t!(stream.read_timeout()));

        assert_eq!(None, t!(stream.write_timeout()));

        t!(stream.set_write_timeout(Some(dur)));
        assert_eq!(Some(dur), t!(stream.write_timeout()));

        t!(stream.set_read_timeout(None));
        assert_eq!(None, t!(stream.read_timeout()));

        t!(stream.set_write_timeout(None));
        assert_eq!(None, t!(stream.write_timeout()));
        drop(listener);
    }

    #[test]
    fn test_read_timeout() {
        let addr = next_test_ip4();
        let listener = t!(TcpListener::bind(&addr));

        let mut stream = t!(TcpStream::connect(&("localhost", addr.port())));
        t!(stream.set_read_timeout(Some(Duration::from_millis(1000))));

        let mut buf = [0; 10];
        let start = Instant::now();
        let kind = stream.read(&mut buf).err().expect("expected error").kind();
        assert!(kind == ErrorKind::WouldBlock || kind == ErrorKind::TimedOut);
        assert!(start.elapsed() > Duration::from_millis(400));
        drop(listener);
    }

    #[test]
    fn test_read_with_timeout() {
        let addr = next_test_ip4();
        let listener = t!(TcpListener::bind(&addr));

        let mut stream = t!(TcpStream::connect(&("localhost", addr.port())));
        t!(stream.set_read_timeout(Some(Duration::from_millis(1000))));

        let mut other_end = t!(listener.accept()).0;
        t!(other_end.write_all(b"hello world"));

        let mut buf = [0; 11];
        t!(stream.read(&mut buf));
        assert_eq!(b"hello world", &buf[..]);

        let start = Instant::now();
        let kind = stream.read(&mut buf).err().expect("expected error").kind();
        assert!(kind == ErrorKind::WouldBlock || kind == ErrorKind::TimedOut);
        assert!(start.elapsed() > Duration::from_millis(400));
        drop(listener);
    }

    #[test]
    fn nodelay() {
        let addr = next_test_ip4();
        let _listener = t!(TcpListener::bind(&addr));

        let stream = t!(TcpStream::connect(&("localhost", addr.port())));

        assert_eq!(false, t!(stream.nodelay()));
        t!(stream.set_nodelay(true));
        assert_eq!(true, t!(stream.nodelay()));
        t!(stream.set_nodelay(false));
        assert_eq!(false, t!(stream.nodelay()));
    }

    #[test]
    fn ttl() {
        let ttl = 100;

        let addr = next_test_ip4();
        let listener = t!(TcpListener::bind(&addr));

        t!(listener.set_ttl(ttl));
        assert_eq!(ttl, t!(listener.ttl()));

        let stream = t!(TcpStream::connect(&("localhost", addr.port())));

        t!(stream.set_ttl(ttl));
        assert_eq!(ttl, t!(stream.ttl()));
    }

    #[test]
    fn set_nonblocking() {
        let addr = next_test_ip4();
        let listener = t!(TcpListener::bind(&addr));

        t!(listener.set_nonblocking(true));
        t!(listener.set_nonblocking(false));

        let mut stream = t!(TcpStream::connect(&("localhost", addr.port())));

        t!(stream.set_nonblocking(false));
        t!(stream.set_nonblocking(true));

        let mut buf = [0];
        match stream.read(&mut buf) {
            Ok(_) => panic!("expected error"),
            Err(ref e) if e.kind() == ErrorKind::WouldBlock => {}
            Err(e) => panic!("unexpected error {}", e),
        }
    }
}
