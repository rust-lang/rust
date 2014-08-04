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
use collections::MutableSeq;
use io::IoResult;
use iter::Iterator;
use slice::ImmutableSlice;
use result::{Ok,Err};
use io::net::addrinfo::get_host_addresses;
use io::net::ip::SocketAddr;
use io::{IoError, ConnectionFailed, InvalidInput};
use io::{Reader, Writer, Listener, Acceptor};
use io::{standard_error, TimedOut};
use from_str::FromStr;
use kinds::Send;
use option::{None, Some, Option};
use boxed::Box;
use rt::rtio::{IoFactory, LocalIo, RtioSocket, RtioTcpListener};
use rt::rtio::{RtioTcpAcceptor, RtioTcpStream};
use rt::rtio;
use time::Duration;

/// A structure which represents a TCP stream between a local socket and a
/// remote socket.
///
/// # Example
///
/// ```no_run
/// # #![allow(unused_must_use)]
/// use std::io::TcpStream;
///
/// let mut stream = TcpStream::connect("127.0.0.1", 34254);
///
/// stream.write([1]);
/// let mut buf = [0];
/// stream.read(buf);
/// drop(stream); // close the connection
/// ```
pub struct TcpStream {
    obj: Box<RtioTcpStream + Send>,
}

impl TcpStream {
    fn new(s: Box<RtioTcpStream + Send>) -> TcpStream {
        TcpStream { obj: s }
    }

    /// Open a TCP connection to a remote host by hostname or IP address.
    ///
    /// `host` can be a hostname or IP address string. If no error is
    /// encountered, then `Ok(stream)` is returned.
    pub fn connect(host: &str, port: u16) -> IoResult<TcpStream> {
        let addresses = match FromStr::from_str(host) {
            Some(addr) => vec!(addr),
            None => try!(get_host_addresses(host))
        };
        let mut err = IoError {
            kind: ConnectionFailed,
            desc: "no addresses found for hostname",
            detail: None
        };
        for addr in addresses.iter() {
            let addr = rtio::SocketAddr{ ip: super::to_rtio(*addr), port: port };
            let result = LocalIo::maybe_raise(|io| {
                io.tcp_connect(addr, None).map(TcpStream::new)
            });
            match result {
                Ok(stream) => {
                    return Ok(stream)
                }
                Err(connect_err) => {
                    err = IoError::from_rtio_error(connect_err)
                }
            }
        }
        Err(err)
    }

    /// Creates a TCP connection to a remote socket address, timing out after
    /// the specified duration.
    ///
    /// This is the same as the `connect` method, except that if the timeout
    /// specified elapses before a connection is made an error will be
    /// returned. The error's kind will be `TimedOut`.
    ///
    /// Note that the `addr` argument may one day be split into a separate host
    /// and port, similar to the API seen in `connect`.
    ///
    /// If a `timeout` with zero or negative duration is specified then
    /// the function returns `Err`, with the error kind set to `TimedOut`.
    #[experimental = "the timeout argument may eventually change types"]
    pub fn connect_timeout(addr: SocketAddr,
                           timeout: Duration) -> IoResult<TcpStream> {
        if timeout <= Duration::milliseconds(0) {
            return Err(standard_error(TimedOut));
        }

        let SocketAddr { ip, port } = addr;
        let addr = rtio::SocketAddr { ip: super::to_rtio(ip), port: port };
        LocalIo::maybe_raise(|io| {
            io.tcp_connect(addr, Some(timeout.num_milliseconds() as u64)).map(TcpStream::new)
        }).map_err(IoError::from_rtio_error)
    }

    /// Returns the socket address of the remote peer of this TCP connection.
    pub fn peer_name(&mut self) -> IoResult<SocketAddr> {
        match self.obj.peer_name() {
            Ok(rtio::SocketAddr { ip, port }) => {
                Ok(SocketAddr { ip: super::from_rtio(ip), port: port })
            }
            Err(e) => Err(IoError::from_rtio_error(e)),
        }
    }

    /// Returns the socket address of the local half of this TCP connection.
    pub fn socket_name(&mut self) -> IoResult<SocketAddr> {
        match self.obj.socket_name() {
            Ok(rtio::SocketAddr { ip, port }) => {
                Ok(SocketAddr { ip: super::from_rtio(ip), port: port })
            }
            Err(e) => Err(IoError::from_rtio_error(e)),
        }
    }

    /// Sets the nodelay flag on this connection to the boolean specified
    #[experimental]
    pub fn set_nodelay(&mut self, nodelay: bool) -> IoResult<()> {
        if nodelay {
            self.obj.nodelay()
        } else {
            self.obj.control_congestion()
        }.map_err(IoError::from_rtio_error)
    }

    /// Sets the keepalive timeout to the timeout specified.
    ///
    /// If the value specified is `None`, then the keepalive flag is cleared on
    /// this connection. Otherwise, the keepalive timeout will be set to the
    /// specified time, in seconds.
    #[experimental]
    pub fn set_keepalive(&mut self, delay_in_seconds: Option<uint>) -> IoResult<()> {
        match delay_in_seconds {
            Some(i) => self.obj.keepalive(i),
            None => self.obj.letdie(),
        }.map_err(IoError::from_rtio_error)
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
    /// let mut stream = TcpStream::connect("127.0.0.1", 34254).unwrap();
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
    /// stream.read(buf);
    /// ```
    ///
    /// Note that this method affects all cloned handles associated with this
    /// stream, not just this one handle.
    pub fn close_read(&mut self) -> IoResult<()> {
        self.obj.close_read().map_err(IoError::from_rtio_error)
    }

    /// Closes the writing half of this connection.
    ///
    /// This method will close the writing portion of this connection, causing
    /// all future writes to immediately return with an error.
    ///
    /// Note that this method affects all cloned handles associated with this
    /// stream, not just this one handle.
    pub fn close_write(&mut self) -> IoResult<()> {
        self.obj.close_write().map_err(IoError::from_rtio_error)
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
        self.obj.set_timeout(timeout_ms)
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
        self.obj.set_read_timeout(timeout_ms)
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
        self.obj.set_write_timeout(timeout_ms)
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
        TcpStream { obj: self.obj.clone() }
    }
}

impl Reader for TcpStream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        self.obj.read(buf).map_err(IoError::from_rtio_error)
    }
}

impl Writer for TcpStream {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        self.obj.write(buf).map_err(IoError::from_rtio_error)
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
/// let listener = TcpListener::bind("127.0.0.1", 80);
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
    obj: Box<RtioTcpListener + Send>,
}

impl TcpListener {
    /// Creates a new `TcpListener` which will be bound to the specified IP
    /// and port. This listener is not ready for accepting connections,
    /// `listen` must be called on it before that's possible.
    ///
    /// Binding with a port number of 0 will request that the OS assigns a port
    /// to this listener. The port allocated can be queried via the
    /// `socket_name` function.
    pub fn bind(addr: &str, port: u16) -> IoResult<TcpListener> {
        match FromStr::from_str(addr) {
            Some(ip) => {
                let addr = rtio::SocketAddr{
                    ip: super::to_rtio(ip),
                    port: port,
                };
                LocalIo::maybe_raise(|io| {
                    io.tcp_bind(addr).map(|l| TcpListener { obj: l })
                }).map_err(IoError::from_rtio_error)
            }
            None => {
                Err(IoError{
                    kind: InvalidInput,
                    desc: "invalid IP address specified",
                    detail: None
                })
            }
        }
    }

    /// Returns the local socket address of this listener.
    pub fn socket_name(&mut self) -> IoResult<SocketAddr> {
        match self.obj.socket_name() {
            Ok(rtio::SocketAddr { ip, port }) => {
                Ok(SocketAddr { ip: super::from_rtio(ip), port: port })
            }
            Err(e) => Err(IoError::from_rtio_error(e)),
        }
    }
}

impl Listener<TcpStream, TcpAcceptor> for TcpListener {
    fn listen(self) -> IoResult<TcpAcceptor> {
        match self.obj.listen() {
            Ok(acceptor) => Ok(TcpAcceptor { obj: acceptor }),
            Err(e) => Err(IoError::from_rtio_error(e)),
        }
    }
}

/// The accepting half of a TCP socket server. This structure is created through
/// a `TcpListener`'s `listen` method, and this object can be used to accept new
/// `TcpStream` instances.
pub struct TcpAcceptor {
    obj: Box<RtioTcpAcceptor + Send>,
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
    /// let mut a = TcpListener::bind("127.0.0.1", 8482).listen().unwrap();
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
    pub fn set_timeout(&mut self, ms: Option<u64>) { self.obj.set_timeout(ms); }

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
    /// let mut a = TcpListener::bind("127.0.0.1", 8482).listen().unwrap();
    /// let a2 = a.clone();
    ///
    /// spawn(proc() {
    ///     let mut a2 = a2;
    ///     for socket in a2.incoming() {
    ///         match socket {
    ///             Ok(s) => { /* handle s */ }
    ///             Err(ref e) if e.kind == EndOfFile => break, // closed
    ///             Err(e) => fail!("unexpected error: {}", e),
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
        self.obj.close_accept().map_err(IoError::from_rtio_error)
    }
}

impl Acceptor<TcpStream> for TcpAcceptor {
    fn accept(&mut self) -> IoResult<TcpStream> {
        match self.obj.accept(){
            Ok(s) => Ok(TcpStream::new(s)),
            Err(e) => Err(IoError::from_rtio_error(e)),
        }
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
        TcpAcceptor { obj: self.obj.clone() }
    }
}

#[cfg(test)]
#[allow(experimental)]
mod test {
    use super::*;
    use io::net::ip::SocketAddr;
    use io::*;
    use prelude::*;

    // FIXME #11530 this fails on android because tests are run as root
    iotest!(fn bind_error() {
        match TcpListener::bind("0.0.0.0", 1) {
            Ok(..) => fail!(),
            Err(e) => assert_eq!(e.kind, PermissionDenied),
        }
    } #[cfg_attr(any(windows, target_os = "android"), ignore)])

    iotest!(fn connect_error() {
        match TcpStream::connect("0.0.0.0", 1) {
            Ok(..) => fail!(),
            Err(e) => assert_eq!(e.kind, ConnectionRefused),
        }
    })

    iotest!(fn listen_ip4_localhost() {
        let socket_addr = next_test_ip4();
        let ip_str = socket_addr.ip.to_string();
        let port = socket_addr.port;
        let listener = TcpListener::bind(ip_str.as_slice(), port);
        let mut acceptor = listener.listen();

        spawn(proc() {
            let mut stream = TcpStream::connect("localhost", port);
            stream.write([144]).unwrap();
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        stream.read(buf).unwrap();
        assert!(buf[0] == 144);
    })

    iotest!(fn connect_localhost() {
        let addr = next_test_ip4();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let mut acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();

        spawn(proc() {
            let mut stream = TcpStream::connect("localhost", addr.port);
            stream.write([64]).unwrap();
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        stream.read(buf).unwrap();
        assert!(buf[0] == 64);
    })

    iotest!(fn connect_ip4_loopback() {
        let addr = next_test_ip4();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let mut acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();

        spawn(proc() {
            let mut stream = TcpStream::connect("127.0.0.1", addr.port);
            stream.write([44]).unwrap();
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        stream.read(buf).unwrap();
        assert!(buf[0] == 44);
    })

    iotest!(fn connect_ip6_loopback() {
        let addr = next_test_ip6();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let mut acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();

        spawn(proc() {
            let mut stream = TcpStream::connect("::1", addr.port);
            stream.write([66]).unwrap();
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        stream.read(buf).unwrap();
        assert!(buf[0] == 66);
    })

    iotest!(fn smoke_test_ip4() {
        let addr = next_test_ip4();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let mut acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();

        spawn(proc() {
            let mut stream = TcpStream::connect(ip_str.as_slice(), port);
            stream.write([99]).unwrap();
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        stream.read(buf).unwrap();
        assert!(buf[0] == 99);
    })

    iotest!(fn smoke_test_ip6() {
        let addr = next_test_ip6();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let mut acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();

        spawn(proc() {
            let mut stream = TcpStream::connect(ip_str.as_slice(), port);
            stream.write([99]).unwrap();
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        stream.read(buf).unwrap();
        assert!(buf[0] == 99);
    })

    iotest!(fn read_eof_ip4() {
        let addr = next_test_ip4();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let mut acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();

        spawn(proc() {
            let _stream = TcpStream::connect(ip_str.as_slice(), port);
            // Close
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        let nread = stream.read(buf);
        assert!(nread.is_err());
    })

    iotest!(fn read_eof_ip6() {
        let addr = next_test_ip6();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let mut acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();

        spawn(proc() {
            let _stream = TcpStream::connect(ip_str.as_slice(), port);
            // Close
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        let nread = stream.read(buf);
        assert!(nread.is_err());
    })

    iotest!(fn read_eof_twice_ip4() {
        let addr = next_test_ip4();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let mut acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();

        spawn(proc() {
            let _stream = TcpStream::connect(ip_str.as_slice(), port);
            // Close
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        let nread = stream.read(buf);
        assert!(nread.is_err());

        match stream.read(buf) {
            Ok(..) => fail!(),
            Err(ref e) => {
                assert!(e.kind == NotConnected || e.kind == EndOfFile,
                        "unknown kind: {}", e.kind);
            }
        }
    })

    iotest!(fn read_eof_twice_ip6() {
        let addr = next_test_ip6();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let mut acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();

        spawn(proc() {
            let _stream = TcpStream::connect(ip_str.as_slice(), port);
            // Close
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        let nread = stream.read(buf);
        assert!(nread.is_err());

        match stream.read(buf) {
            Ok(..) => fail!(),
            Err(ref e) => {
                assert!(e.kind == NotConnected || e.kind == EndOfFile,
                        "unknown kind: {}", e.kind);
            }
        }
    })

    iotest!(fn write_close_ip4() {
        let addr = next_test_ip4();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let mut acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();

        spawn(proc() {
            let _stream = TcpStream::connect(ip_str.as_slice(), port);
            // Close
        });

        let mut stream = acceptor.accept();
        let buf = [0];
        loop {
            match stream.write(buf) {
                Ok(..) => {}
                Err(e) => {
                    assert!(e.kind == ConnectionReset ||
                            e.kind == BrokenPipe ||
                            e.kind == ConnectionAborted,
                            "unknown error: {}", e);
                    break;
                }
            }
        }
    })

    iotest!(fn write_close_ip6() {
        let addr = next_test_ip6();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let mut acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();

        spawn(proc() {
            let _stream = TcpStream::connect(ip_str.as_slice(), port);
            // Close
        });

        let mut stream = acceptor.accept();
        let buf = [0];
        loop {
            match stream.write(buf) {
                Ok(..) => {}
                Err(e) => {
                    assert!(e.kind == ConnectionReset ||
                            e.kind == BrokenPipe ||
                            e.kind == ConnectionAborted,
                            "unknown error: {}", e);
                    break;
                }
            }
        }
    })

    iotest!(fn multiple_connect_serial_ip4() {
        let addr = next_test_ip4();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let max = 10u;
        let mut acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();

        spawn(proc() {
            for _ in range(0, max) {
                let mut stream = TcpStream::connect(ip_str.as_slice(), port);
                stream.write([99]).unwrap();
            }
        });

        for ref mut stream in acceptor.incoming().take(max) {
            let mut buf = [0];
            stream.read(buf).unwrap();
            assert_eq!(buf[0], 99);
        }
    })

    iotest!(fn multiple_connect_serial_ip6() {
        let addr = next_test_ip6();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let max = 10u;
        let mut acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();

        spawn(proc() {
            for _ in range(0, max) {
                let mut stream = TcpStream::connect(ip_str.as_slice(), port);
                stream.write([99]).unwrap();
            }
        });

        for ref mut stream in acceptor.incoming().take(max) {
            let mut buf = [0];
            stream.read(buf).unwrap();
            assert_eq!(buf[0], 99);
        }
    })

    iotest!(fn multiple_connect_interleaved_greedy_schedule_ip4() {
        let addr = next_test_ip4();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        static MAX: int = 10;
        let acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();

        spawn(proc() {
            let mut acceptor = acceptor;
            for (i, stream) in acceptor.incoming().enumerate().take(MAX as uint) {
                // Start another task to handle the connection
                spawn(proc() {
                    let mut stream = stream;
                    let mut buf = [0];
                    stream.read(buf).unwrap();
                    assert!(buf[0] == i as u8);
                    debug!("read");
                });
            }
        });

        connect(0, addr);

        fn connect(i: int, addr: SocketAddr) {
            let ip_str = addr.ip.to_string();
            let port = addr.port;
            if i == MAX { return }

            spawn(proc() {
                debug!("connecting");
                let mut stream = TcpStream::connect(ip_str.as_slice(), port);
                // Connect again before writing
                connect(i + 1, addr);
                debug!("writing");
                stream.write([i as u8]).unwrap();
            });
        }
    })

    iotest!(fn multiple_connect_interleaved_greedy_schedule_ip6() {
        let addr = next_test_ip6();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        static MAX: int = 10;
        let acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();

        spawn(proc() {
            let mut acceptor = acceptor;
            for (i, stream) in acceptor.incoming().enumerate().take(MAX as uint) {
                // Start another task to handle the connection
                spawn(proc() {
                    let mut stream = stream;
                    let mut buf = [0];
                    stream.read(buf).unwrap();
                    assert!(buf[0] == i as u8);
                    debug!("read");
                });
            }
        });

        connect(0, addr);

        fn connect(i: int, addr: SocketAddr) {
            let ip_str = addr.ip.to_string();
            let port = addr.port;
            if i == MAX { return }

            spawn(proc() {
                debug!("connecting");
                let mut stream = TcpStream::connect(ip_str.as_slice(), port);
                // Connect again before writing
                connect(i + 1, addr);
                debug!("writing");
                stream.write([i as u8]).unwrap();
            });
        }
    })

    iotest!(fn multiple_connect_interleaved_lazy_schedule_ip4() {
        static MAX: int = 10;
        let addr = next_test_ip4();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();

        spawn(proc() {
            let mut acceptor = acceptor;
            for stream in acceptor.incoming().take(MAX as uint) {
                // Start another task to handle the connection
                spawn(proc() {
                    let mut stream = stream;
                    let mut buf = [0];
                    stream.read(buf).unwrap();
                    assert!(buf[0] == 99);
                    debug!("read");
                });
            }
        });

        connect(0, addr);

        fn connect(i: int, addr: SocketAddr) {
            let ip_str = addr.ip.to_string();
            let port = addr.port;
            if i == MAX { return }

            spawn(proc() {
                debug!("connecting");
                let mut stream = TcpStream::connect(ip_str.as_slice(), port);
                // Connect again before writing
                connect(i + 1, addr);
                debug!("writing");
                stream.write([99]).unwrap();
            });
        }
    })

    iotest!(fn multiple_connect_interleaved_lazy_schedule_ip6() {
        static MAX: int = 10;
        let addr = next_test_ip6();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();

        spawn(proc() {
            let mut acceptor = acceptor;
            for stream in acceptor.incoming().take(MAX as uint) {
                // Start another task to handle the connection
                spawn(proc() {
                    let mut stream = stream;
                    let mut buf = [0];
                    stream.read(buf).unwrap();
                    assert!(buf[0] == 99);
                    debug!("read");
                });
            }
        });

        connect(0, addr);

        fn connect(i: int, addr: SocketAddr) {
            let ip_str = addr.ip.to_string();
            let port = addr.port;
            if i == MAX { return }

            spawn(proc() {
                debug!("connecting");
                let mut stream = TcpStream::connect(ip_str.as_slice(), port);
                // Connect again before writing
                connect(i + 1, addr);
                debug!("writing");
                stream.write([99]).unwrap();
            });
        }
    })

    pub fn socket_name(addr: SocketAddr) {
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let mut listener = TcpListener::bind(ip_str.as_slice(), port).unwrap();

        // Make sure socket_name gives
        // us the socket we binded to.
        let so_name = listener.socket_name();
        assert!(so_name.is_ok());
        assert_eq!(addr, so_name.unwrap());
    }

    pub fn peer_name(addr: SocketAddr) {
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();
        spawn(proc() {
            let mut acceptor = acceptor;
            acceptor.accept().unwrap();
        });

        let stream = TcpStream::connect(ip_str.as_slice(), port);

        assert!(stream.is_ok());
        let mut stream = stream.unwrap();

        // Make sure peer_name gives us the
        // address/port of the peer we've
        // connected to.
        let peer_name = stream.peer_name();
        assert!(peer_name.is_ok());
        assert_eq!(addr, peer_name.unwrap());
    }

    iotest!(fn socket_and_peer_name_ip4() {
        peer_name(next_test_ip4());
        socket_name(next_test_ip4());
    })

    iotest!(fn socket_and_peer_name_ip6() {
        // FIXME: peer name is not consistent
        //peer_name(next_test_ip6());
        socket_name(next_test_ip6());
    })

    iotest!(fn partial_read() {
        let addr = next_test_ip4();
        let port = addr.port;
        let (tx, rx) = channel();
        spawn(proc() {
            let ip_str = addr.ip.to_string();
            let mut srv = TcpListener::bind(ip_str.as_slice(), port).listen().unwrap();
            tx.send(());
            let mut cl = srv.accept().unwrap();
            cl.write([10]).unwrap();
            let mut b = [0];
            cl.read(b).unwrap();
            tx.send(());
        });

        rx.recv();
        let ip_str = addr.ip.to_string();
        let mut c = TcpStream::connect(ip_str.as_slice(), port).unwrap();
        let mut b = [0, ..10];
        assert_eq!(c.read(b), Ok(1));
        c.write([1]).unwrap();
        rx.recv();
    })

    iotest!(fn double_bind() {
        let addr = next_test_ip4();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let listener = TcpListener::bind(ip_str.as_slice(), port).unwrap().listen();
        assert!(listener.is_ok());
        match TcpListener::bind(ip_str.as_slice(), port).listen() {
            Ok(..) => fail!(),
            Err(e) => {
                assert!(e.kind == ConnectionRefused || e.kind == OtherIoError,
                        "unknown error: {} {}", e, e.kind);
            }
        }
    })

    iotest!(fn fast_rebind() {
        let addr = next_test_ip4();
        let port = addr.port;
        let (tx, rx) = channel();

        spawn(proc() {
            let ip_str = addr.ip.to_string();
            rx.recv();
            let _stream = TcpStream::connect(ip_str.as_slice(), port).unwrap();
            // Close
            rx.recv();
        });

        {
            let ip_str = addr.ip.to_string();
            let mut acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();
            tx.send(());
            {
                let _stream = acceptor.accept().unwrap();
                // Close client
                tx.send(());
            }
            // Close listener
        }
        let _listener = TcpListener::bind(addr.ip.to_string().as_slice(), port);
    })

    iotest!(fn tcp_clone_smoke() {
        let addr = next_test_ip4();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let mut acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();

        spawn(proc() {
            let mut s = TcpStream::connect(ip_str.as_slice(), port);
            let mut buf = [0, 0];
            assert_eq!(s.read(buf), Ok(1));
            assert_eq!(buf[0], 1);
            s.write([2]).unwrap();
        });

        let mut s1 = acceptor.accept().unwrap();
        let s2 = s1.clone();

        let (tx1, rx1) = channel();
        let (tx2, rx2) = channel();
        spawn(proc() {
            let mut s2 = s2;
            rx1.recv();
            s2.write([1]).unwrap();
            tx2.send(());
        });
        tx1.send(());
        let mut buf = [0, 0];
        assert_eq!(s1.read(buf), Ok(1));
        rx2.recv();
    })

    iotest!(fn tcp_clone_two_read() {
        let addr = next_test_ip6();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let mut acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();
        let (tx1, rx) = channel();
        let tx2 = tx1.clone();

        spawn(proc() {
            let mut s = TcpStream::connect(ip_str.as_slice(), port);
            s.write([1]).unwrap();
            rx.recv();
            s.write([2]).unwrap();
            rx.recv();
        });

        let mut s1 = acceptor.accept().unwrap();
        let s2 = s1.clone();

        let (done, rx) = channel();
        spawn(proc() {
            let mut s2 = s2;
            let mut buf = [0, 0];
            s2.read(buf).unwrap();
            tx2.send(());
            done.send(());
        });
        let mut buf = [0, 0];
        s1.read(buf).unwrap();
        tx1.send(());

        rx.recv();
    })

    iotest!(fn tcp_clone_two_write() {
        let addr = next_test_ip4();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let mut acceptor = TcpListener::bind(ip_str.as_slice(), port).listen();

        spawn(proc() {
            let mut s = TcpStream::connect(ip_str.as_slice(), port);
            let mut buf = [0, 1];
            s.read(buf).unwrap();
            s.read(buf).unwrap();
        });

        let mut s1 = acceptor.accept().unwrap();
        let s2 = s1.clone();

        let (done, rx) = channel();
        spawn(proc() {
            let mut s2 = s2;
            s2.write([1]).unwrap();
            done.send(());
        });
        s1.write([2]).unwrap();

        rx.recv();
    })

    iotest!(fn shutdown_smoke() {
        use rt::rtio::RtioTcpStream;

        let addr = next_test_ip4();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let a = TcpListener::bind(ip_str.as_slice(), port).unwrap().listen();
        spawn(proc() {
            let mut a = a;
            let mut c = a.accept().unwrap();
            assert_eq!(c.read_to_end(), Ok(vec!()));
            c.write([1]).unwrap();
        });

        let mut s = TcpStream::connect(ip_str.as_slice(), port).unwrap();
        assert!(s.obj.close_write().is_ok());
        assert!(s.write([1]).is_err());
        assert_eq!(s.read_to_end(), Ok(vec!(1)));
    })

    iotest!(fn accept_timeout() {
        let addr = next_test_ip4();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let mut a = TcpListener::bind(ip_str.as_slice(), port).unwrap().listen().unwrap();

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
                tx.send(TcpStream::connect(addr.ip.to_string().as_slice(),
                                           port).unwrap());
            });
            let _l = rx.recv();
            for i in range(0i, 1001) {
                match a.accept() {
                    Ok(..) => break,
                    Err(ref e) if e.kind == TimedOut => {}
                    Err(e) => fail!("error: {}", e),
                }
                ::task::deschedule();
                if i == 1000 { fail!("should have a pending connection") }
            }
        }

        // Unset the timeout and make sure that this always blocks.
        a.set_timeout(None);
        spawn(proc() {
            drop(TcpStream::connect(addr.ip.to_string().as_slice(),
                                    port).unwrap());
        });
        a.accept().unwrap();
    })

    iotest!(fn close_readwrite_smoke() {
        let addr = next_test_ip4();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let a = TcpListener::bind(ip_str.as_slice(), port).listen().unwrap();
        let (_tx, rx) = channel::<()>();
        spawn(proc() {
            let mut a = a;
            let _s = a.accept().unwrap();
            let _ = rx.recv_opt();
        });

        let mut b = [0];
        let mut s = TcpStream::connect(ip_str.as_slice(), port).unwrap();
        let mut s2 = s.clone();

        // closing should prevent reads/writes
        s.close_write().unwrap();
        assert!(s.write([0]).is_err());
        s.close_read().unwrap();
        assert!(s.read(b).is_err());

        // closing should affect previous handles
        assert!(s2.write([0]).is_err());
        assert!(s2.read(b).is_err());

        // closing should affect new handles
        let mut s3 = s.clone();
        assert!(s3.write([0]).is_err());
        assert!(s3.read(b).is_err());

        // make sure these don't die
        let _ = s2.close_read();
        let _ = s2.close_write();
        let _ = s3.close_read();
        let _ = s3.close_write();
    })

    iotest!(fn close_read_wakes_up() {
        let addr = next_test_ip4();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let a = TcpListener::bind(ip_str.as_slice(), port).listen().unwrap();
        let (_tx, rx) = channel::<()>();
        spawn(proc() {
            let mut a = a;
            let _s = a.accept().unwrap();
            let _ = rx.recv_opt();
        });

        let mut s = TcpStream::connect(ip_str.as_slice(), port).unwrap();
        let s2 = s.clone();
        let (tx, rx) = channel();
        spawn(proc() {
            let mut s2 = s2;
            assert!(s2.read([0]).is_err());
            tx.send(());
        });
        // this should wake up the child task
        s.close_read().unwrap();

        // this test will never finish if the child doesn't wake up
        rx.recv();
    })

    iotest!(fn readwrite_timeouts() {
        let addr = next_test_ip6();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let mut a = TcpListener::bind(ip_str.as_slice(), port).listen().unwrap();
        let (tx, rx) = channel::<()>();
        spawn(proc() {
            let mut s = TcpStream::connect(ip_str.as_slice(), port).unwrap();
            rx.recv();
            assert!(s.write([0]).is_ok());
            let _ = rx.recv_opt();
        });

        let mut s = a.accept().unwrap();
        s.set_timeout(Some(20));
        assert_eq!(s.read([0]).err().unwrap().kind, TimedOut);
        assert_eq!(s.read([0]).err().unwrap().kind, TimedOut);

        s.set_timeout(Some(20));
        for i in range(0i, 1001) {
            match s.write([0, .. 128 * 1024]) {
                Ok(()) | Err(IoError { kind: ShortWrite(..), .. }) => {},
                Err(IoError { kind: TimedOut, .. }) => break,
                Err(e) => fail!("{}", e),
           }
           if i == 1000 { fail!("should have filled up?!"); }
        }
        assert_eq!(s.write([0]).err().unwrap().kind, TimedOut);

        tx.send(());
        s.set_timeout(None);
        assert_eq!(s.read([0, 0]), Ok(1));
    })

    iotest!(fn read_timeouts() {
        let addr = next_test_ip6();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let mut a = TcpListener::bind(ip_str.as_slice(), port).listen().unwrap();
        let (tx, rx) = channel::<()>();
        spawn(proc() {
            let mut s = TcpStream::connect(ip_str.as_slice(), port).unwrap();
            rx.recv();
            let mut amt = 0;
            while amt < 100 * 128 * 1024 {
                match s.read([0, ..128 * 1024]) {
                    Ok(n) => { amt += n; }
                    Err(e) => fail!("{}", e),
                }
            }
            let _ = rx.recv_opt();
        });

        let mut s = a.accept().unwrap();
        s.set_read_timeout(Some(20));
        assert_eq!(s.read([0]).err().unwrap().kind, TimedOut);
        assert_eq!(s.read([0]).err().unwrap().kind, TimedOut);

        tx.send(());
        for _ in range(0i, 100) {
            assert!(s.write([0, ..128 * 1024]).is_ok());
        }
    })

    iotest!(fn write_timeouts() {
        let addr = next_test_ip6();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let mut a = TcpListener::bind(ip_str.as_slice(), port).listen().unwrap();
        let (tx, rx) = channel::<()>();
        spawn(proc() {
            let mut s = TcpStream::connect(ip_str.as_slice(), port).unwrap();
            rx.recv();
            assert!(s.write([0]).is_ok());
            let _ = rx.recv_opt();
        });

        let mut s = a.accept().unwrap();
        s.set_write_timeout(Some(20));
        for i in range(0i, 1001) {
            match s.write([0, .. 128 * 1024]) {
                Ok(()) | Err(IoError { kind: ShortWrite(..), .. }) => {},
                Err(IoError { kind: TimedOut, .. }) => break,
                Err(e) => fail!("{}", e),
           }
           if i == 1000 { fail!("should have filled up?!"); }
        }
        assert_eq!(s.write([0]).err().unwrap().kind, TimedOut);

        tx.send(());
        assert!(s.read([0]).is_ok());
    })

    iotest!(fn timeout_concurrent_read() {
        let addr = next_test_ip6();
        let ip_str = addr.ip.to_string();
        let port = addr.port;
        let mut a = TcpListener::bind(ip_str.as_slice(), port).listen().unwrap();
        let (tx, rx) = channel::<()>();
        spawn(proc() {
            let mut s = TcpStream::connect(ip_str.as_slice(), port).unwrap();
            rx.recv();
            assert_eq!(s.write([0]), Ok(()));
            let _ = rx.recv_opt();
        });

        let mut s = a.accept().unwrap();
        let s2 = s.clone();
        let (tx2, rx2) = channel();
        spawn(proc() {
            let mut s2 = s2;
            assert_eq!(s2.read([0]), Ok(1));
            tx2.send(());
        });

        s.set_read_timeout(Some(20));
        assert_eq!(s.read([0]).err().unwrap().kind, TimedOut);
        tx.send(());

        rx2.recv();
    })

    iotest!(fn clone_while_reading() {
        let addr = next_test_ip6();
        let listen = TcpListener::bind(addr.ip.to_string().as_slice(), addr.port);
        let mut accept = listen.listen().unwrap();

        // Enqueue a task to write to a socket
        let (tx, rx) = channel();
        let (txdone, rxdone) = channel();
        let txdone2 = txdone.clone();
        spawn(proc() {
            let mut tcp = TcpStream::connect(addr.ip.to_string().as_slice(),
                                             addr.port).unwrap();
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
    })

    iotest!(fn clone_accept_smoke() {
        let addr = next_test_ip4();
        let l = TcpListener::bind(addr.ip.to_string().as_slice(), addr.port);
        let mut a = l.listen().unwrap();
        let mut a2 = a.clone();

        spawn(proc() {
            let _ = TcpStream::connect(addr.ip.to_string().as_slice(), addr.port);
        });
        spawn(proc() {
            let _ = TcpStream::connect(addr.ip.to_string().as_slice(), addr.port);
        });

        assert!(a.accept().is_ok());
        assert!(a2.accept().is_ok());
    })

    iotest!(fn clone_accept_concurrent() {
        let addr = next_test_ip4();
        let l = TcpListener::bind(addr.ip.to_string().as_slice(), addr.port);
        let a = l.listen().unwrap();
        let a2 = a.clone();

        let (tx, rx) = channel();
        let tx2 = tx.clone();

        spawn(proc() { let mut a = a; tx.send(a.accept()) });
        spawn(proc() { let mut a = a2; tx2.send(a.accept()) });

        spawn(proc() {
            let _ = TcpStream::connect(addr.ip.to_string().as_slice(), addr.port);
        });
        spawn(proc() {
            let _ = TcpStream::connect(addr.ip.to_string().as_slice(), addr.port);
        });

        assert!(rx.recv().is_ok());
        assert!(rx.recv().is_ok());
    })

    iotest!(fn close_accept_smoke() {
        let addr = next_test_ip4();
        let l = TcpListener::bind(addr.ip.to_string().as_slice(), addr.port);
        let mut a = l.listen().unwrap();

        a.close_accept().unwrap();
        assert_eq!(a.accept().err().unwrap().kind, EndOfFile);
    })

    iotest!(fn close_accept_concurrent() {
        let addr = next_test_ip4();
        let l = TcpListener::bind(addr.ip.to_string().as_slice(), addr.port);
        let a = l.listen().unwrap();
        let mut a2 = a.clone();

        let (tx, rx) = channel();
        spawn(proc() {
            let mut a = a;
            tx.send(a.accept());
        });
        a2.close_accept().unwrap();

        assert_eq!(rx.recv().err().unwrap().kind, EndOfFile);
    })
}
