// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! TCP network connections
//!
//! This module contains the ability to open a TCP stream to a socket address,
//! as well as creating a socket server to accept incoming connections. The
//! destination and binding addresses can either be an IPv4 or IPv6 address.
//!
//! A TCP connection implements the `Reader` and `Writer` traits, while the TCP
//! listener (socket server) implements the `Listener` and `Acceptor` traits.

use clone::Clone;
use io::IoResult;
use iter::Iterator;
use result::Err;
use io::net::ip::{SocketAddr, ToSocketAddr};
use io::{Reader, Writer, Listener, Acceptor};
use io::{standard_error, TimedOut};
use option::{None, Some, Option};
use time::Duration;

use sys::tcp::TcpStream as TcpStreamImp;
use sys::tcp::TcpListener as TcpListenerImp;
use sys::tcp::TcpAcceptor as TcpAcceptorImp;

/// A structure which represents a TCP stream between a local socket and a
/// remote socket.
///
/// # Example
///
/// ```no_run
/// # #![allow(unused_must_use)]
/// use std::io::TcpStream;
///
/// let mut stream = TcpStream::connect("127.0.0.1:34254");
///
/// stream.write(&[1]);
/// let mut buf = [0];
/// stream.read(&mut buf);
/// drop(stream); // close the connection
/// ```
pub struct TcpStream {
    inner: TcpStreamImp,
}

impl TcpStream {
    fn new(s: TcpStreamImp) -> TcpStream { unimplemented!() }

    /// Open a TCP connection to a remote host.
    ///
    /// `addr` is an address of the remote host. Anything which implements `ToSocketAddr`
    /// trait can be supplied for the address; see this trait documentation for
    /// concrete examples.
    pub fn connect<A: ToSocketAddr>(addr: A) -> IoResult<TcpStream> { unimplemented!() }

    /// Creates a TCP connection to a remote socket address, timing out after
    /// the specified duration.
    ///
    /// This is the same as the `connect` method, except that if the timeout
    /// specified elapses before a connection is made an error will be
    /// returned. The error's kind will be `TimedOut`.
    ///
    /// Same as the `connect` method, `addr` argument type can be anything which
    /// implements `ToSocketAddr` trait.
    ///
    /// If a `timeout` with zero or negative duration is specified then
    /// the function returns `Err`, with the error kind set to `TimedOut`.
    #[experimental = "the timeout argument may eventually change types"]
    pub fn connect_timeout<A: ToSocketAddr>(addr: A,
                                            timeout: Duration) -> IoResult<TcpStream> { unimplemented!() }

    /// Returns the socket address of the remote peer of this TCP connection.
    pub fn peer_name(&mut self) -> IoResult<SocketAddr> { unimplemented!() }

    /// Returns the socket address of the local half of this TCP connection.
    pub fn socket_name(&mut self) -> IoResult<SocketAddr> { unimplemented!() }

    /// Sets the nodelay flag on this connection to the boolean specified
    #[experimental]
    pub fn set_nodelay(&mut self, nodelay: bool) -> IoResult<()> { unimplemented!() }

    /// Sets the keepalive timeout to the timeout specified.
    ///
    /// If the value specified is `None`, then the keepalive flag is cleared on
    /// this connection. Otherwise, the keepalive timeout will be set to the
    /// specified time, in seconds.
    #[experimental]
    pub fn set_keepalive(&mut self, delay_in_seconds: Option<uint>) -> IoResult<()> { unimplemented!() }

    /// Closes the reading half of this connection.
    ///
    /// This method will close the reading portion of this connection, causing
    /// all pending and future reads to immediately return with an error.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #![allow(unused_must_use)]
    /// use std::io::timer;
    /// use std::io::TcpStream;
    /// use std::time::Duration;
    ///
    /// let mut stream = TcpStream::connect("127.0.0.1:34254").unwrap();
    /// let stream2 = stream.clone();
    ///
    /// spawn(proc() {
    ///     // close this stream after one second
    ///     timer::sleep(Duration::seconds(1));
    ///     let mut stream = stream2;
    ///     stream.close_read();
    /// });
    ///
    /// // wait for some data, will get canceled after one second
    /// let mut buf = [0];
    /// stream.read(&mut buf);
    /// ```
    ///
    /// Note that this method affects all cloned handles associated with this
    /// stream, not just this one handle.
    pub fn close_read(&mut self) -> IoResult<()> { unimplemented!() }

    /// Closes the writing half of this connection.
    ///
    /// This method will close the writing portion of this connection, causing
    /// all future writes to immediately return with an error.
    ///
    /// Note that this method affects all cloned handles associated with this
    /// stream, not just this one handle.
    pub fn close_write(&mut self) -> IoResult<()> { unimplemented!() }

    /// Sets a timeout, in milliseconds, for blocking operations on this stream.
    ///
    /// This function will set a timeout for all blocking operations (including
    /// reads and writes) on this stream. The timeout specified is a relative
    /// time, in milliseconds, into the future after which point operations will
    /// time out. This means that the timeout must be reset periodically to keep
    /// it from expiring. Specifying a value of `None` will clear the timeout
    /// for this stream.
    ///
    /// The timeout on this stream is local to this stream only. Setting a
    /// timeout does not affect any other cloned instances of this stream, nor
    /// does the timeout propagated to cloned handles of this stream. Setting
    /// this timeout will override any specific read or write timeouts
    /// previously set for this stream.
    ///
    /// For clarification on the semantics of interrupting a read and a write,
    /// take a look at `set_read_timeout` and `set_write_timeout`.
    #[experimental = "the timeout argument may change in type and value"]
    pub fn set_timeout(&mut self, timeout_ms: Option<u64>) { unimplemented!() }

    /// Sets the timeout for read operations on this stream.
    ///
    /// See documentation in `set_timeout` for the semantics of this read time.
    /// This will overwrite any previous read timeout set through either this
    /// function or `set_timeout`.
    ///
    /// # Errors
    ///
    /// When this timeout expires, if there is no pending read operation, no
    /// action is taken. Otherwise, the read operation will be scheduled to
    /// promptly return. If a timeout error is returned, then no data was read
    /// during the timeout period.
    #[experimental = "the timeout argument may change in type and value"]
    pub fn set_read_timeout(&mut self, timeout_ms: Option<u64>) { unimplemented!() }

    /// Sets the timeout for write operations on this stream.
    ///
    /// See documentation in `set_timeout` for the semantics of this write time.
    /// This will overwrite any previous write timeout set through either this
    /// function or `set_timeout`.
    ///
    /// # Errors
    ///
    /// When this timeout expires, if there is no pending write operation, no
    /// action is taken. Otherwise, the pending write operation will be
    /// scheduled to promptly return. The actual state of the underlying stream
    /// is not specified.
    ///
    /// The write operation may return an error of type `ShortWrite` which
    /// indicates that the object is known to have written an exact number of
    /// bytes successfully during the timeout period, and the remaining bytes
    /// were never written.
    ///
    /// If the write operation returns `TimedOut`, then it the timeout primitive
    /// does not know how many bytes were written as part of the timeout
    /// operation. It may be the case that bytes continue to be written in an
    /// asynchronous fashion after the call to write returns.
    #[experimental = "the timeout argument may change in type and value"]
    pub fn set_write_timeout(&mut self, timeout_ms: Option<u64>) { unimplemented!() }
}

impl Clone for TcpStream {
    /// Creates a new handle to this TCP stream, allowing for simultaneous reads
    /// and writes of this connection.
    ///
    /// The underlying TCP stream will not be closed until all handles to the
    /// stream have been deallocated. All handles will also follow the same
    /// stream, but two concurrent reads will not receive the same data.
    /// Instead, the first read will receive the first packet received, and the
    /// second read will receive the second packet.
    fn clone(&self) -> TcpStream { unimplemented!() }
}

impl Reader for TcpStream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { unimplemented!() }
}

impl Writer for TcpStream {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { unimplemented!() }
}

/// A structure representing a socket server. This listener is used to create a
/// `TcpAcceptor` which can be used to accept sockets on a local port.
///
/// # Example
///
/// ```rust
/// # fn main() { }
/// # fn foo() {
/// # #![allow(dead_code)]
/// use std::io::{TcpListener, TcpStream};
/// use std::io::{Acceptor, Listener};
///
/// let listener = TcpListener::bind("127.0.0.1:80");
///
/// // bind the listener to the specified address
/// let mut acceptor = listener.listen();
///
/// fn handle_client(mut stream: TcpStream) {
///     // ...
/// # &mut stream; // silence unused mutability/variable warning
/// }
/// // accept connections and process them, spawning a new tasks for each one
/// for stream in acceptor.incoming() {
///     match stream {
///         Err(e) => { /* connection failed */ }
///         Ok(stream) => spawn(proc() {
///             // connection succeeded
///             handle_client(stream)
///         })
///     }
/// }
///
/// // close the socket server
/// drop(acceptor);
/// # }
/// ```
pub struct TcpListener {
    inner: TcpListenerImp,
}

impl TcpListener {
    /// Creates a new `TcpListener` which will be bound to the specified address.
    /// This listener is not ready for accepting connections, `listen` must be called
    /// on it before that's possible.
    ///
    /// Binding with a port number of 0 will request that the OS assigns a port
    /// to this listener. The port allocated can be queried via the
    /// `socket_name` function.
    ///
    /// The address type can be any implementor of `ToSocketAddr` trait. See its
    /// documentation for concrete examples.
    pub fn bind<A: ToSocketAddr>(addr: A) -> IoResult<TcpListener> { unimplemented!() }

    /// Returns the local socket address of this listener.
    pub fn socket_name(&mut self) -> IoResult<SocketAddr> { unimplemented!() }
}

impl Listener<TcpStream, TcpAcceptor> for TcpListener {
    fn listen(self) -> IoResult<TcpAcceptor> { unimplemented!() }
}

/// The accepting half of a TCP socket server. This structure is created through
/// a `TcpListener`'s `listen` method, and this object can be used to accept new
/// `TcpStream` instances.
pub struct TcpAcceptor {
    inner: TcpAcceptorImp,
}

impl TcpAcceptor {
    /// Prevents blocking on all future accepts after `ms` milliseconds have
    /// elapsed.
    ///
    /// This function is used to set a deadline after which this acceptor will
    /// time out accepting any connections. The argument is the relative
    /// distance, in milliseconds, to a point in the future after which all
    /// accepts will fail.
    ///
    /// If the argument specified is `None`, then any previously registered
    /// timeout is cleared.
    ///
    /// A timeout of `0` can be used to "poll" this acceptor to see if it has
    /// any pending connections. All pending connections will be accepted,
    /// regardless of whether the timeout has expired or not (the accept will
    /// not block in this case).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #![allow(experimental)]
    /// use std::io::TcpListener;
    /// use std::io::{Listener, Acceptor, TimedOut};
    ///
    /// let mut a = TcpListener::bind("127.0.0.1:8482").listen().unwrap();
    ///
    /// // After 100ms have passed, all accepts will fail
    /// a.set_timeout(Some(100));
    ///
    /// match a.accept() {
    ///     Ok(..) => println!("accepted a socket"),
    ///     Err(ref e) if e.kind == TimedOut => { println!("timed out!"); }
    ///     Err(e) => println!("err: {}", e),
    /// }
    ///
    /// // Reset the timeout and try again
    /// a.set_timeout(Some(100));
    /// let socket = a.accept();
    ///
    /// // Clear the timeout and block indefinitely waiting for a connection
    /// a.set_timeout(None);
    /// let socket = a.accept();
    /// ```
    #[experimental = "the type of the argument and name of this function are \
                      subject to change"]
    pub fn set_timeout(&mut self, ms: Option<u64>) { unimplemented!() }

    /// Closes the accepting capabilities of this acceptor.
    ///
    /// This function is similar to `TcpStream`'s `close_{read,write}` methods
    /// in that it will affect *all* cloned handles of this acceptor's original
    /// handle.
    ///
    /// Once this function succeeds, all future calls to `accept` will return
    /// immediately with an error, preventing all future calls to accept. The
    /// underlying socket will not be relinquished back to the OS until all
    /// acceptors have been deallocated.
    ///
    /// This is useful for waking up a thread in an accept loop to indicate that
    /// it should exit.
    ///
    /// # Example
    ///
    /// ```
    /// # #![allow(experimental)]
    /// use std::io::{TcpListener, Listener, Acceptor, EndOfFile};
    ///
    /// let mut a = TcpListener::bind("127.0.0.1:8482").listen().unwrap();
    /// let a2 = a.clone();
    ///
    /// spawn(proc() {
    ///     let mut a2 = a2;
    ///     for socket in a2.incoming() {
    ///         match socket {
    ///             Ok(s) => { /* handle s */ }
    ///             Err(ref e) if e.kind == EndOfFile => break, // closed
    ///             Err(e) => panic!("unexpected error: {}", e),
    ///         }
    ///     }
    /// });
    ///
    /// # fn wait_for_sigint() {}
    /// // Now that our accept loop is running, wait for the program to be
    /// // requested to exit.
    /// wait_for_sigint();
    ///
    /// // Signal our accept loop to exit
    /// assert!(a.close_accept().is_ok());
    /// ```
    #[experimental]
    pub fn close_accept(&mut self) -> IoResult<()> { unimplemented!() }
}

impl Acceptor<TcpStream> for TcpAcceptor {
    fn accept(&mut self) -> IoResult<TcpStream> { unimplemented!() }
}

impl Clone for TcpAcceptor {
    /// Creates a new handle to this TCP acceptor, allowing for simultaneous
    /// accepts.
    ///
    /// The underlying TCP acceptor will not be closed until all handles to the
    /// acceptor have been deallocated. Incoming connections will be received on
    /// at most once acceptor, the same connection will not be accepted twice.
    ///
    /// The `close_accept` method will shut down *all* acceptors cloned from the
    /// same original acceptor, whereas the `set_timeout` method only affects
    /// the selector that it is called on.
    ///
    /// This function is useful for creating a handle to invoke `close_accept`
    /// on to wake up any other task blocked in `accept`.
    fn clone(&self) -> TcpAcceptor { unimplemented!() }
}
