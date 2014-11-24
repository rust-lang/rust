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
/// The socket will be closed when the value is dropped.
///
/// # Example
///
/// ```no_run
/// use std::io::TcpStream;
///
/// {
///     let mut stream = TcpStream::connect("127.0.0.1:34254");
///
///     // ignore the Result
///     let _ = stream.write(&[1]);
///
///     let mut buf = [0];
///     let _ = stream.read(&mut buf); // ignore here too
/// } // the stream is closed here
/// ```
pub struct TcpStream {
    inner: TcpStreamImp,
}

impl TcpStream {
    fn new(s: TcpStreamImp) -> TcpStream {
        TcpStream { inner: s }
    }

    /// Open a TCP connection to a remote host.
    ///
    /// `addr` is an address of the remote host. Anything which implements `ToSocketAddr`
    /// trait can be supplied for the address; see this trait documentation for
    /// concrete examples.
    pub fn connect<A: ToSocketAddr>(addr: A) -> IoResult<TcpStream> {
        super::with_addresses(addr, |addr| {
            TcpStreamImp::connect(addr, None).map(TcpStream::new)
        })
    }

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
                                            timeout: Duration) -> IoResult<TcpStream> {
        if timeout <= Duration::milliseconds(0) {
            return Err(standard_error(TimedOut));
        }

        super::with_addresses(addr, |addr| {
            TcpStreamImp::connect(addr, Some(timeout.num_milliseconds() as u64))
                .map(TcpStream::new)
        })
    }

    /// Returns the socket address of the remote peer of this TCP connection.
    pub fn peer_name(&mut self) -> IoResult<SocketAddr> {
        self.inner.peer_name()
    }

    /// Returns the socket address of the local half of this TCP connection.
    pub fn socket_name(&mut self) -> IoResult<SocketAddr> {
        self.inner.socket_name()
    }

    /// Sets the nodelay flag on this connection to the boolean specified
    #[experimental]
    pub fn set_nodelay(&mut self, nodelay: bool) -> IoResult<()> {
        self.inner.set_nodelay(nodelay)
    }

    /// Sets the keepalive timeout to the timeout specified.
    ///
    /// If the value specified is `None`, then the keepalive flag is cleared on
    /// this connection. Otherwise, the keepalive timeout will be set to the
    /// specified time, in seconds.
    #[experimental]
    pub fn set_keepalive(&mut self, delay_in_seconds: Option<uint>) -> IoResult<()> {
        self.inner.set_keepalive(delay_in_seconds)
    }

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
    pub fn close_read(&mut self) -> IoResult<()> {
        self.inner.close_read()
    }

    /// Closes the writing half of this connection.
    ///
    /// This method will close the writing portion of this connection, causing
    /// all future writes to immediately return with an error.
    ///
    /// Note that this method affects all cloned handles associated with this
    /// stream, not just this one handle.
    pub fn close_write(&mut self) -> IoResult<()> {
        self.inner.close_write()
    }

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
    pub fn set_timeout(&mut self, timeout_ms: Option<u64>) {
        self.inner.set_timeout(timeout_ms)
    }

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
    pub fn set_read_timeout(&mut self, timeout_ms: Option<u64>) {
        self.inner.set_read_timeout(timeout_ms)
    }

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
    pub fn set_write_timeout(&mut self, timeout_ms: Option<u64>) {
        self.inner.set_write_timeout(timeout_ms)
    }
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
    fn clone(&self) -> TcpStream {
        TcpStream { inner: self.inner.clone() }
    }
}

impl Reader for TcpStream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        self.inner.read(buf)
    }
}

impl Writer for TcpStream {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        self.inner.write(buf)
    }
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
    pub fn bind<A: ToSocketAddr>(addr: A) -> IoResult<TcpListener> {
        super::with_addresses(addr, |addr| {
            TcpListenerImp::bind(addr).map(|inner| TcpListener { inner: inner })
        })
    }

    /// Returns the local socket address of this listener.
    pub fn socket_name(&mut self) -> IoResult<SocketAddr> {
        self.inner.socket_name()
    }
}

impl Listener<TcpStream, TcpAcceptor> for TcpListener {
    fn listen(self) -> IoResult<TcpAcceptor> {
        self.inner.listen(128).map(|a| TcpAcceptor { inner: a })
    }
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
    pub fn set_timeout(&mut self, ms: Option<u64>) { self.inner.set_timeout(ms); }

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
    pub fn close_accept(&mut self) -> IoResult<()> {
        self.inner.close_accept()
    }
}

impl Acceptor<TcpStream> for TcpAcceptor {
    fn accept(&mut self) -> IoResult<TcpStream> {
        self.inner.accept().map(TcpStream::new)
    }
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
    fn clone(&self) -> TcpAcceptor {
        TcpAcceptor { inner: self.inner.clone() }
    }
}

#[cfg(test)]
#[allow(experimental)]
mod test {
    use io::net::tcp::*;
    use io::net::ip::*;
    use io::*;
    use io::test::*;
    use prelude::*;

    // FIXME #11530 this fails on android because tests are run as root
    #[cfg_attr(any(windows, target_os = "android"), ignore)]
    #[test]
    fn bind_error() {
        match TcpListener::bind("0.0.0.0:1") {
            Ok(..) => panic!(),
            Err(e) => assert_eq!(e.kind, PermissionDenied),
        }
    }

    #[test]
    fn connect_error() {
        match TcpStream::connect("0.0.0.0:1") {
            Ok(..) => panic!(),
            Err(e) => assert_eq!(e.kind, ConnectionRefused),
        }
    }

    #[test]
    fn listen_ip4_localhost() {
        let socket_addr = next_test_ip4();
        let listener = TcpListener::bind(socket_addr);
        let mut acceptor = listener.listen();

        spawn(proc() {
            let mut stream = TcpStream::connect(("localhost", socket_addr.port));
            stream.write(&[144]).unwrap();
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        stream.read(&mut buf).unwrap();
        assert!(buf[0] == 144);
    }

    #[test]
    fn connect_localhost() {
        let addr = next_test_ip4();
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let mut stream = TcpStream::connect(("localhost", addr.port));
            stream.write(&[64]).unwrap();
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        stream.read(&mut buf).unwrap();
        assert!(buf[0] == 64);
    }

    #[test]
    fn connect_ip4_loopback() {
        let addr = next_test_ip4();
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let mut stream = TcpStream::connect(("127.0.0.1", addr.port));
            stream.write(&[44]).unwrap();
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        stream.read(&mut buf).unwrap();
        assert!(buf[0] == 44);
    }

    #[test]
    fn connect_ip6_loopback() {
        let addr = next_test_ip6();
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let mut stream = TcpStream::connect(("::1", addr.port));
            stream.write(&[66]).unwrap();
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        stream.read(&mut buf).unwrap();
        assert!(buf[0] == 66);
    }

    #[test]
    fn smoke_test_ip4() {
        let addr = next_test_ip4();
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let mut stream = TcpStream::connect(addr);
            stream.write(&[99]).unwrap();
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        stream.read(&mut buf).unwrap();
        assert!(buf[0] == 99);
    }

    #[test]
    fn smoke_test_ip6() {
        let addr = next_test_ip6();
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let mut stream = TcpStream::connect(addr);
            stream.write(&[99]).unwrap();
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        stream.read(&mut buf).unwrap();
        assert!(buf[0] == 99);
    }

    #[test]
    fn read_eof_ip4() {
        let addr = next_test_ip4();
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let _stream = TcpStream::connect(addr);
            // Close
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        let nread = stream.read(&mut buf);
        assert!(nread.is_err());
    }

    #[test]
    fn read_eof_ip6() {
        let addr = next_test_ip6();
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let _stream = TcpStream::connect(addr);
            // Close
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        let nread = stream.read(&mut buf);
        assert!(nread.is_err());
    }

    #[test]
    fn read_eof_twice_ip4() {
        let addr = next_test_ip4();
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let _stream = TcpStream::connect(addr);
            // Close
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        let nread = stream.read(&mut buf);
        assert!(nread.is_err());

        match stream.read(&mut buf) {
            Ok(..) => panic!(),
            Err(ref e) => {
                assert!(e.kind == NotConnected || e.kind == EndOfFile,
                        "unknown kind: {}", e.kind);
            }
        }
    }

    #[test]
    fn read_eof_twice_ip6() {
        let addr = next_test_ip6();
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let _stream = TcpStream::connect(addr);
            // Close
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        let nread = stream.read(&mut buf);
        assert!(nread.is_err());

        match stream.read(&mut buf) {
            Ok(..) => panic!(),
            Err(ref e) => {
                assert!(e.kind == NotConnected || e.kind == EndOfFile,
                        "unknown kind: {}", e.kind);
            }
        }
    }

    #[test]
    fn write_close_ip4() {
        let addr = next_test_ip4();
        let mut acceptor = TcpListener::bind(addr).listen();

        let (tx, rx) = channel();
        spawn(proc() {
            drop(TcpStream::connect(addr));
            tx.send(());
        });

        let mut stream = acceptor.accept();
        rx.recv();
        let buf = [0];
        match stream.write(&buf) {
            Ok(..) => {}
            Err(e) => {
                assert!(e.kind == ConnectionReset ||
                        e.kind == BrokenPipe ||
                        e.kind == ConnectionAborted,
                        "unknown error: {}", e);
            }
        }
    }

    #[test]
    fn write_close_ip6() {
        let addr = next_test_ip6();
        let mut acceptor = TcpListener::bind(addr).listen();

        let (tx, rx) = channel();
        spawn(proc() {
            drop(TcpStream::connect(addr));
            tx.send(());
        });

        let mut stream = acceptor.accept();
        rx.recv();
        let buf = [0];
        match stream.write(&buf) {
            Ok(..) => {}
            Err(e) => {
                assert!(e.kind == ConnectionReset ||
                        e.kind == BrokenPipe ||
                        e.kind == ConnectionAborted,
                        "unknown error: {}", e);
            }
        }
    }

    #[test]
    fn multiple_connect_serial_ip4() {
        let addr = next_test_ip4();
        let max = 10u;
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            for _ in range(0, max) {
                let mut stream = TcpStream::connect(addr);
                stream.write(&[99]).unwrap();
            }
        });

        for ref mut stream in acceptor.incoming().take(max) {
            let mut buf = [0];
            stream.read(&mut buf).unwrap();
            assert_eq!(buf[0], 99);
        }
    }

    #[test]
    fn multiple_connect_serial_ip6() {
        let addr = next_test_ip6();
        let max = 10u;
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            for _ in range(0, max) {
                let mut stream = TcpStream::connect(addr);
                stream.write(&[99]).unwrap();
            }
        });

        for ref mut stream in acceptor.incoming().take(max) {
            let mut buf = [0];
            stream.read(&mut buf).unwrap();
            assert_eq!(buf[0], 99);
        }
    }

    #[test]
    fn multiple_connect_interleaved_greedy_schedule_ip4() {
        let addr = next_test_ip4();
        static MAX: int = 10;
        let acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let mut acceptor = acceptor;
            for (i, stream) in acceptor.incoming().enumerate().take(MAX as uint) {
                // Start another task to handle the connection
                spawn(proc() {
                    let mut stream = stream;
                    let mut buf = [0];
                    stream.read(&mut buf).unwrap();
                    assert!(buf[0] == i as u8);
                    debug!("read");
                });
            }
        });

        connect(0, addr);

        fn connect(i: int, addr: SocketAddr) {
            if i == MAX { return }

            spawn(proc() {
                debug!("connecting");
                let mut stream = TcpStream::connect(addr);
                // Connect again before writing
                connect(i + 1, addr);
                debug!("writing");
                stream.write(&[i as u8]).unwrap();
            });
        }
    }

    #[test]
    fn multiple_connect_interleaved_greedy_schedule_ip6() {
        let addr = next_test_ip6();
        static MAX: int = 10;
        let acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let mut acceptor = acceptor;
            for (i, stream) in acceptor.incoming().enumerate().take(MAX as uint) {
                // Start another task to handle the connection
                spawn(proc() {
                    let mut stream = stream;
                    let mut buf = [0];
                    stream.read(&mut buf).unwrap();
                    assert!(buf[0] == i as u8);
                    debug!("read");
                });
            }
        });

        connect(0, addr);

        fn connect(i: int, addr: SocketAddr) {
            if i == MAX { return }

            spawn(proc() {
                debug!("connecting");
                let mut stream = TcpStream::connect(addr);
                // Connect again before writing
                connect(i + 1, addr);
                debug!("writing");
                stream.write(&[i as u8]).unwrap();
            });
        }
    }

    #[test]
    fn multiple_connect_interleaved_lazy_schedule_ip4() {
        static MAX: int = 10;
        let addr = next_test_ip4();
        let acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let mut acceptor = acceptor;
            for stream in acceptor.incoming().take(MAX as uint) {
                // Start another task to handle the connection
                spawn(proc() {
                    let mut stream = stream;
                    let mut buf = [0];
                    stream.read(&mut buf).unwrap();
                    assert!(buf[0] == 99);
                    debug!("read");
                });
            }
        });

        connect(0, addr);

        fn connect(i: int, addr: SocketAddr) {
            if i == MAX { return }

            spawn(proc() {
                debug!("connecting");
                let mut stream = TcpStream::connect(addr);
                // Connect again before writing
                connect(i + 1, addr);
                debug!("writing");
                stream.write(&[99]).unwrap();
            });
        }
    }

    #[test]
    fn multiple_connect_interleaved_lazy_schedule_ip6() {
        static MAX: int = 10;
        let addr = next_test_ip6();
        let acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let mut acceptor = acceptor;
            for stream in acceptor.incoming().take(MAX as uint) {
                // Start another task to handle the connection
                spawn(proc() {
                    let mut stream = stream;
                    let mut buf = [0];
                    stream.read(&mut buf).unwrap();
                    assert!(buf[0] == 99);
                    debug!("read");
                });
            }
        });

        connect(0, addr);

        fn connect(i: int, addr: SocketAddr) {
            if i == MAX { return }

            spawn(proc() {
                debug!("connecting");
                let mut stream = TcpStream::connect(addr);
                // Connect again before writing
                connect(i + 1, addr);
                debug!("writing");
                stream.write(&[99]).unwrap();
            });
        }
    }

    pub fn socket_name(addr: SocketAddr) {
        let mut listener = TcpListener::bind(addr).unwrap();

        // Make sure socket_name gives
        // us the socket we binded to.
        let so_name = listener.socket_name();
        assert!(so_name.is_ok());
        assert_eq!(addr, so_name.unwrap());
    }

    pub fn peer_name(addr: SocketAddr) {
        let acceptor = TcpListener::bind(addr).listen();
        spawn(proc() {
            let mut acceptor = acceptor;
            acceptor.accept().unwrap();
        });

        let stream = TcpStream::connect(addr);

        assert!(stream.is_ok());
        let mut stream = stream.unwrap();

        // Make sure peer_name gives us the
        // address/port of the peer we've
        // connected to.
        let peer_name = stream.peer_name();
        assert!(peer_name.is_ok());
        assert_eq!(addr, peer_name.unwrap());
    }

    #[test]
    fn socket_and_peer_name_ip4() {
        peer_name(next_test_ip4());
        socket_name(next_test_ip4());
    }

    #[test]
    fn socket_and_peer_name_ip6() {
        // FIXME: peer name is not consistent
        //peer_name(next_test_ip6());
        socket_name(next_test_ip6());
    }

    #[test]
    fn partial_read() {
        let addr = next_test_ip4();
        let (tx, rx) = channel();
        spawn(proc() {
            let mut srv = TcpListener::bind(addr).listen().unwrap();
            tx.send(());
            let mut cl = srv.accept().unwrap();
            cl.write(&[10]).unwrap();
            let mut b = [0];
            cl.read(&mut b).unwrap();
            tx.send(());
        });

        rx.recv();
        let mut c = TcpStream::connect(addr).unwrap();
        let mut b = [0, ..10];
        assert_eq!(c.read(&mut b), Ok(1));
        c.write(&[1]).unwrap();
        rx.recv();
    }

    #[test]
    fn double_bind() {
        let addr = next_test_ip4();
        let listener = TcpListener::bind(addr).unwrap().listen();
        assert!(listener.is_ok());
        match TcpListener::bind(addr).listen() {
            Ok(..) => panic!(),
            Err(e) => {
                assert!(e.kind == ConnectionRefused || e.kind == OtherIoError,
                        "unknown error: {} {}", e, e.kind);
            }
        }
    }

    #[test]
    fn fast_rebind() {
        let addr = next_test_ip4();
        let (tx, rx) = channel();

        spawn(proc() {
            rx.recv();
            let _stream = TcpStream::connect(addr).unwrap();
            // Close
            rx.recv();
        });

        {
            let mut acceptor = TcpListener::bind(addr).listen();
            tx.send(());
            {
                let _stream = acceptor.accept().unwrap();
                // Close client
                tx.send(());
            }
            // Close listener
        }
        let _listener = TcpListener::bind(addr);
    }

    #[test]
    fn tcp_clone_smoke() {
        let addr = next_test_ip4();
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let mut s = TcpStream::connect(addr);
            let mut buf = [0, 0];
            assert_eq!(s.read(&mut buf), Ok(1));
            assert_eq!(buf[0], 1);
            s.write(&[2]).unwrap();
        });

        let mut s1 = acceptor.accept().unwrap();
        let s2 = s1.clone();

        let (tx1, rx1) = channel();
        let (tx2, rx2) = channel();
        spawn(proc() {
            let mut s2 = s2;
            rx1.recv();
            s2.write(&[1]).unwrap();
            tx2.send(());
        });
        tx1.send(());
        let mut buf = [0, 0];
        assert_eq!(s1.read(&mut buf), Ok(1));
        rx2.recv();
    }

    #[test]
    fn tcp_clone_two_read() {
        let addr = next_test_ip6();
        let mut acceptor = TcpListener::bind(addr).listen();
        let (tx1, rx) = channel();
        let tx2 = tx1.clone();

        spawn(proc() {
            let mut s = TcpStream::connect(addr);
            s.write(&[1]).unwrap();
            rx.recv();
            s.write(&[2]).unwrap();
            rx.recv();
        });

        let mut s1 = acceptor.accept().unwrap();
        let s2 = s1.clone();

        let (done, rx) = channel();
        spawn(proc() {
            let mut s2 = s2;
            let mut buf = [0, 0];
            s2.read(&mut buf).unwrap();
            tx2.send(());
            done.send(());
        });
        let mut buf = [0, 0];
        s1.read(&mut buf).unwrap();
        tx1.send(());

        rx.recv();
    }

    #[test]
    fn tcp_clone_two_write() {
        let addr = next_test_ip4();
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let mut s = TcpStream::connect(addr);
            let mut buf = [0, 1];
            s.read(&mut buf).unwrap();
            s.read(&mut buf).unwrap();
        });

        let mut s1 = acceptor.accept().unwrap();
        let s2 = s1.clone();

        let (done, rx) = channel();
        spawn(proc() {
            let mut s2 = s2;
            s2.write(&[1]).unwrap();
            done.send(());
        });
        s1.write(&[2]).unwrap();

        rx.recv();
    }

    #[test]
    fn shutdown_smoke() {
        let addr = next_test_ip4();
        let a = TcpListener::bind(addr).unwrap().listen();
        spawn(proc() {
            let mut a = a;
            let mut c = a.accept().unwrap();
            assert_eq!(c.read_to_end(), Ok(vec!()));
            c.write(&[1]).unwrap();
        });

        let mut s = TcpStream::connect(addr).unwrap();
        assert!(s.inner.close_write().is_ok());
        assert!(s.write(&[1]).is_err());
        assert_eq!(s.read_to_end(), Ok(vec!(1)));
    }

    #[test]
    fn accept_timeout() {
        let addr = next_test_ip4();
        let mut a = TcpListener::bind(addr).unwrap().listen().unwrap();

        a.set_timeout(Some(10));

        // Make sure we time out once and future invocations also time out
        let err = a.accept().err().unwrap();
        assert_eq!(err.kind, TimedOut);
        let err = a.accept().err().unwrap();
        assert_eq!(err.kind, TimedOut);

        // Also make sure that even though the timeout is expired that we will
        // continue to receive any pending connections.
        //
        // FIXME: freebsd apparently never sees the pending connection, but
        //        testing manually always works. Need to investigate this
        //        flakiness.
        if !cfg!(target_os = "freebsd") {
            let (tx, rx) = channel();
            spawn(proc() {
                tx.send(TcpStream::connect(addr).unwrap());
            });
            let _l = rx.recv();
            for i in range(0i, 1001) {
                match a.accept() {
                    Ok(..) => break,
                    Err(ref e) if e.kind == TimedOut => {}
                    Err(e) => panic!("error: {}", e),
                }
                ::task::deschedule();
                if i == 1000 { panic!("should have a pending connection") }
            }
        }

        // Unset the timeout and make sure that this always blocks.
        a.set_timeout(None);
        spawn(proc() {
            drop(TcpStream::connect(addr).unwrap());
        });
        a.accept().unwrap();
    }

    #[test]
    fn close_readwrite_smoke() {
        let addr = next_test_ip4();
        let a = TcpListener::bind(addr).listen().unwrap();
        let (_tx, rx) = channel::<()>();
        spawn(proc() {
            let mut a = a;
            let _s = a.accept().unwrap();
            let _ = rx.recv_opt();
        });

        let mut b = [0];
        let mut s = TcpStream::connect(addr).unwrap();
        let mut s2 = s.clone();

        // closing should prevent reads/writes
        s.close_write().unwrap();
        assert!(s.write(&[0]).is_err());
        s.close_read().unwrap();
        assert!(s.read(&mut b).is_err());

        // closing should affect previous handles
        assert!(s2.write(&[0]).is_err());
        assert!(s2.read(&mut b).is_err());

        // closing should affect new handles
        let mut s3 = s.clone();
        assert!(s3.write(&[0]).is_err());
        assert!(s3.read(&mut b).is_err());

        // make sure these don't die
        let _ = s2.close_read();
        let _ = s2.close_write();
        let _ = s3.close_read();
        let _ = s3.close_write();
    }

    #[test]
    fn close_read_wakes_up() {
        let addr = next_test_ip4();
        let a = TcpListener::bind(addr).listen().unwrap();
        let (_tx, rx) = channel::<()>();
        spawn(proc() {
            let mut a = a;
            let _s = a.accept().unwrap();
            let _ = rx.recv_opt();
        });

        let mut s = TcpStream::connect(addr).unwrap();
        let s2 = s.clone();
        let (tx, rx) = channel();
        spawn(proc() {
            let mut s2 = s2;
            assert!(s2.read(&mut [0]).is_err());
            tx.send(());
        });
        // this should wake up the child task
        s.close_read().unwrap();

        // this test will never finish if the child doesn't wake up
        rx.recv();
    }

    #[test]
    fn readwrite_timeouts() {
        let addr = next_test_ip6();
        let mut a = TcpListener::bind(addr).listen().unwrap();
        let (tx, rx) = channel::<()>();
        spawn(proc() {
            let mut s = TcpStream::connect(addr).unwrap();
            rx.recv();
            assert!(s.write(&[0]).is_ok());
            let _ = rx.recv_opt();
        });

        let mut s = a.accept().unwrap();
        s.set_timeout(Some(20));
        assert_eq!(s.read(&mut [0]).err().unwrap().kind, TimedOut);
        assert_eq!(s.read(&mut [0]).err().unwrap().kind, TimedOut);

        s.set_timeout(Some(20));
        for i in range(0i, 1001) {
            match s.write(&[0, .. 128 * 1024]) {
                Ok(()) | Err(IoError { kind: ShortWrite(..), .. }) => {},
                Err(IoError { kind: TimedOut, .. }) => break,
                Err(e) => panic!("{}", e),
           }
           if i == 1000 { panic!("should have filled up?!"); }
        }
        assert_eq!(s.write(&[0]).err().unwrap().kind, TimedOut);

        tx.send(());
        s.set_timeout(None);
        assert_eq!(s.read(&mut [0, 0]), Ok(1));
    }

    #[test]
    fn read_timeouts() {
        let addr = next_test_ip6();
        let mut a = TcpListener::bind(addr).listen().unwrap();
        let (tx, rx) = channel::<()>();
        spawn(proc() {
            let mut s = TcpStream::connect(addr).unwrap();
            rx.recv();
            let mut amt = 0;
            while amt < 100 * 128 * 1024 {
                match s.read(&mut [0, ..128 * 1024]) {
                    Ok(n) => { amt += n; }
                    Err(e) => panic!("{}", e),
                }
            }
            let _ = rx.recv_opt();
        });

        let mut s = a.accept().unwrap();
        s.set_read_timeout(Some(20));
        assert_eq!(s.read(&mut [0]).err().unwrap().kind, TimedOut);
        assert_eq!(s.read(&mut [0]).err().unwrap().kind, TimedOut);

        tx.send(());
        for _ in range(0i, 100) {
            assert!(s.write(&[0, ..128 * 1024]).is_ok());
        }
    }

    #[test]
    fn write_timeouts() {
        let addr = next_test_ip6();
        let mut a = TcpListener::bind(addr).listen().unwrap();
        let (tx, rx) = channel::<()>();
        spawn(proc() {
            let mut s = TcpStream::connect(addr).unwrap();
            rx.recv();
            assert!(s.write(&[0]).is_ok());
            let _ = rx.recv_opt();
        });

        let mut s = a.accept().unwrap();
        s.set_write_timeout(Some(20));
        for i in range(0i, 1001) {
            match s.write(&[0, .. 128 * 1024]) {
                Ok(()) | Err(IoError { kind: ShortWrite(..), .. }) => {},
                Err(IoError { kind: TimedOut, .. }) => break,
                Err(e) => panic!("{}", e),
           }
           if i == 1000 { panic!("should have filled up?!"); }
        }
        assert_eq!(s.write(&[0]).err().unwrap().kind, TimedOut);

        tx.send(());
        assert!(s.read(&mut [0]).is_ok());
    }

    #[test]
    fn timeout_concurrent_read() {
        let addr = next_test_ip6();
        let mut a = TcpListener::bind(addr).listen().unwrap();
        let (tx, rx) = channel::<()>();
        spawn(proc() {
            let mut s = TcpStream::connect(addr).unwrap();
            rx.recv();
            assert_eq!(s.write(&[0]), Ok(()));
            let _ = rx.recv_opt();
        });

        let mut s = a.accept().unwrap();
        let s2 = s.clone();
        let (tx2, rx2) = channel();
        spawn(proc() {
            let mut s2 = s2;
            assert_eq!(s2.read(&mut [0]), Ok(1));
            tx2.send(());
        });

        s.set_read_timeout(Some(20));
        assert_eq!(s.read(&mut [0]).err().unwrap().kind, TimedOut);
        tx.send(());

        rx2.recv();
    }

    #[test]
    fn clone_while_reading() {
        let addr = next_test_ip6();
        let listen = TcpListener::bind(addr);
        let mut accept = listen.listen().unwrap();

        // Enqueue a task to write to a socket
        let (tx, rx) = channel();
        let (txdone, rxdone) = channel();
        let txdone2 = txdone.clone();
        spawn(proc() {
            let mut tcp = TcpStream::connect(addr).unwrap();
            rx.recv();
            tcp.write_u8(0).unwrap();
            txdone2.send(());
        });

        // Spawn off a reading clone
        let tcp = accept.accept().unwrap();
        let tcp2 = tcp.clone();
        let txdone3 = txdone.clone();
        spawn(proc() {
            let mut tcp2 = tcp2;
            tcp2.read_u8().unwrap();
            txdone3.send(());
        });

        // Try to ensure that the reading clone is indeed reading
        for _ in range(0i, 50) {
            ::task::deschedule();
        }

        // clone the handle again while it's reading, then let it finish the
        // read.
        let _ = tcp.clone();
        tx.send(());
        rxdone.recv();
        rxdone.recv();
    }

    #[test]
    fn clone_accept_smoke() {
        let addr = next_test_ip4();
        let l = TcpListener::bind(addr);
        let mut a = l.listen().unwrap();
        let mut a2 = a.clone();

        spawn(proc() {
            let _ = TcpStream::connect(addr);
        });
        spawn(proc() {
            let _ = TcpStream::connect(addr);
        });

        assert!(a.accept().is_ok());
        assert!(a2.accept().is_ok());
    }

    #[test]
    fn clone_accept_concurrent() {
        let addr = next_test_ip4();
        let l = TcpListener::bind(addr);
        let a = l.listen().unwrap();
        let a2 = a.clone();

        let (tx, rx) = channel();
        let tx2 = tx.clone();

        spawn(proc() { let mut a = a; tx.send(a.accept()) });
        spawn(proc() { let mut a = a2; tx2.send(a.accept()) });

        spawn(proc() {
            let _ = TcpStream::connect(addr);
        });
        spawn(proc() {
            let _ = TcpStream::connect(addr);
        });

        assert!(rx.recv().is_ok());
        assert!(rx.recv().is_ok());
    }

    #[test]
    fn close_accept_smoke() {
        let addr = next_test_ip4();
        let l = TcpListener::bind(addr);
        let mut a = l.listen().unwrap();

        a.close_accept().unwrap();
        assert_eq!(a.accept().err().unwrap().kind, EndOfFile);
    }

    #[test]
    fn close_accept_concurrent() {
        let addr = next_test_ip4();
        let l = TcpListener::bind(addr);
        let a = l.listen().unwrap();
        let mut a2 = a.clone();

        let (tx, rx) = channel();
        spawn(proc() {
            let mut a = a;
            tx.send(a.accept());
        });
        a2.close_accept().unwrap();

        assert_eq!(rx.recv().err().unwrap().kind, EndOfFile);
    }
}
