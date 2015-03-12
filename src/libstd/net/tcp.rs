// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;
use io::prelude::*;

use io;
use net::{ToSocketAddrs, SocketAddr, Shutdown};
use sys_common::net2 as net_imp;
use sys_common::AsInner;

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
pub struct TcpStream(net_imp::TcpStream);

/// A structure representing a socket server.
///
/// # Examples
///
/// ```no_run
/// use std::net::{TcpListener, TcpStream};
/// use std::thread;
///
/// let listener = TcpListener::bind("127.0.0.1:80").unwrap();
///
/// fn handle_client(stream: TcpStream) {
///     // ...
/// }
///
/// // accept connections and process them, spawning a new thread for each one
/// for stream in listener.incoming() {
///     match stream {
///         Ok(stream) => {
///             thread::spawn(move|| {
///                 // connection succeeded
///                 handle_client(stream)
///             });
///         }
///         Err(e) => { /* connection failed */ }
///     }
/// }
///
/// // close the socket server
/// drop(listener);
/// ```
pub struct TcpListener(net_imp::TcpListener);

/// An infinite iterator over the connections from a `TcpListener`.
///
/// This iterator will infinitely yield `Some` of the accepted connections. It
/// is equivalent to calling `accept` in a loop.
pub struct Incoming<'a> { listener: &'a TcpListener }

impl TcpStream {
    /// Open a TCP connection to a remote host.
    ///
    /// `addr` is an address of the remote host. Anything which implements
    /// `ToSocketAddrs` trait can be supplied for the address; see this trait
    /// documentation for concrete examples.
    pub fn connect<A: ToSocketAddrs + ?Sized>(addr: &A) -> io::Result<TcpStream> {
        super::each_addr(addr, net_imp::TcpStream::connect).map(TcpStream)
    }

    /// Returns the socket address of the remote peer of this TCP connection.
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        self.0.peer_addr()
    }

    /// Returns the socket address of the local half of this TCP connection.
    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        self.0.socket_addr()
    }

    /// Shut down the read, write, or both halves of this connection.
    ///
    /// This function will cause all pending and future I/O on the specified
    /// portions to return immediately with an appropriate value (see the
    /// documentation of `Shutdown`).
    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        self.0.shutdown(how)
    }

    /// Create a new independently owned handle to the underlying socket.
    ///
    /// The returned `TcpStream` is a reference to the same stream that this
    /// object references. Both handles will read and write the same stream of
    /// data, and options set on one stream will be propagated to the other
    /// stream.
    pub fn try_clone(&self) -> io::Result<TcpStream> {
        self.0.duplicate().map(TcpStream)
    }

    /// Sets the nodelay flag on this connection to the boolean specified
    pub fn set_nodelay(&self, nodelay: bool) -> io::Result<()> {
        self.0.set_nodelay(nodelay)
    }

    /// Sets the keepalive timeout to the timeout specified.
    ///
    /// If the value specified is `None`, then the keepalive flag is cleared on
    /// this connection. Otherwise, the keepalive timeout will be set to the
    /// specified time, in seconds.
    pub fn set_keepalive(&self, seconds: Option<u32>) -> io::Result<()> {
        self.0.set_keepalive(seconds)
    }
}

impl Read for TcpStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> { self.0.read(buf) }
}
impl Write for TcpStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> { self.0.write(buf) }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}
impl<'a> Read for &'a TcpStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> { self.0.read(buf) }
}
impl<'a> Write for &'a TcpStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> { self.0.write(buf) }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

impl AsInner<net_imp::TcpStream> for TcpStream {
    fn as_inner(&self) -> &net_imp::TcpStream { &self.0 }
}

impl TcpListener {
    /// Creates a new `TcpListener` which will be bound to the specified
    /// address.
    ///
    /// The returned listener is ready for accepting connections.
    ///
    /// Binding with a port number of 0 will request that the OS assigns a port
    /// to this listener. The port allocated can be queried via the
    /// `socket_addr` function.
    ///
    /// The address type can be any implementer of `ToSocketAddrs` trait. See
    /// its documentation for concrete examples.
    pub fn bind<A: ToSocketAddrs + ?Sized>(addr: &A) -> io::Result<TcpListener> {
        super::each_addr(addr, net_imp::TcpListener::bind).map(TcpListener)
    }

    /// Returns the local socket address of this listener.
    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        self.0.socket_addr()
    }

    /// Create a new independently owned handle to the underlying socket.
    ///
    /// The returned `TcpListener` is a reference to the same socket that this
    /// object references. Both handles can be used to accept incoming
    /// connections and options set on one listener will affect the other.
    pub fn try_clone(&self) -> io::Result<TcpListener> {
        self.0.duplicate().map(TcpListener)
    }

    /// Accept a new incoming connection from this listener.
    ///
    /// This function will block the calling thread until a new TCP connection
    /// is established. When established, the corresponding `TcpStream` and the
    /// remote peer's address will be returned.
    pub fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        self.0.accept().map(|(a, b)| (TcpStream(a), b))
    }

    /// Returns an iterator over the connections being received on this
    /// listener.
    ///
    /// The returned iterator will never returned `None` and will also not yield
    /// the peer's `SocketAddr` structure.
    pub fn incoming(&self) -> Incoming {
        Incoming { listener: self }
    }
}

impl<'a> Iterator for Incoming<'a> {
    type Item = io::Result<TcpStream>;
    fn next(&mut self) -> Option<io::Result<TcpStream>> {
        Some(self.listener.accept().map(|p| p.0))
    }
}

impl AsInner<net_imp::TcpListener> for TcpListener {
    fn as_inner(&self) -> &net_imp::TcpListener { &self.0 }
}

#[cfg(test)]
mod tests {
    use prelude::v1::*;

    use io::ErrorKind;
    use io::prelude::*;
    use net::*;
    use net::test::{next_test_ip4, next_test_ip6};
    use sync::mpsc::channel;
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
                // EADDRNOTAVAIL is mapped to ConnectionRefused
                assert_eq!(e.kind(), ErrorKind::ConnectionRefused),
        }
    }

    #[test]
    fn connect_error() {
        match TcpStream::connect("0.0.0.0:1") {
            Ok(..) => panic!(),
            Err(e) => assert!((e.kind() == ErrorKind::ConnectionRefused)
                              || (e.kind() == ErrorKind::InvalidInput)),
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
    fn connect_ip4_loopback() {
        let addr = next_test_ip4();
        let acceptor = t!(TcpListener::bind(&addr));

        let _t = thread::spawn(move|| {
            let mut stream = t!(TcpStream::connect(&("127.0.0.1", addr.port())));
            t!(stream.write(&[44]));
        });

        let mut stream = t!(acceptor.accept()).0;
        let mut buf = [0];
        t!(stream.read(&mut buf));
        assert!(buf[0] == 44);
    }

    #[test]
    fn connect_ip6_loopback() {
        let addr = next_test_ip6();
        let acceptor = t!(TcpListener::bind(&addr));

        let _t = thread::spawn(move|| {
            let mut stream = t!(TcpStream::connect(&("::1", addr.port())));
            t!(stream.write(&[66]));
        });

        let mut stream = t!(acceptor.accept()).0;
        let mut buf = [0];
        t!(stream.read(&mut buf));
        assert!(buf[0] == 66);
    }

    #[test]
    fn smoke_test_ip6() {
        each_ip(&mut |addr| {
            let acceptor = t!(TcpListener::bind(&addr));

            let (tx, rx) = channel();
            let _t = thread::spawn(move|| {
                let mut stream = t!(TcpStream::connect(&addr));
                t!(stream.write(&[99]));
                tx.send(t!(stream.socket_addr())).unwrap();
            });

            let (mut stream, addr) = t!(acceptor.accept());
            let mut buf = [0];
            t!(stream.read(&mut buf));
            assert!(buf[0] == 99);
            assert_eq!(addr, t!(rx.recv()));
        })
    }

    #[test]
    fn read_eof_ip4() {
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
    fn multiple_connect_serial_ip4() {
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
        static MAX: usize = 10;
        each_ip(&mut |addr| {
            let acceptor = t!(TcpListener::bind(&addr));

            let _t = thread::spawn(move|| {
                let acceptor = acceptor;
                for (i, stream) in acceptor.incoming().enumerate().take(MAX) {
                    // Start another task to handle the connection
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
    fn multiple_connect_interleaved_lazy_schedule_ip4() {
        const MAX: usize = 10;
        each_ip(&mut |addr| {
            let acceptor = t!(TcpListener::bind(&addr));

            let _t = thread::spawn(move|| {
                for stream in acceptor.incoming().take(MAX) {
                    // Start another task to handle the connection
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
    fn socket_and_peer_name_ip4() {
        each_ip(&mut |addr| {
            let listener = t!(TcpListener::bind(&addr));
            let so_name = t!(listener.socket_addr());
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
            assert_eq!(c.read(&mut b), Ok(1));
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
                            e.kind() == ErrorKind::Other,
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
                assert_eq!(s.read(&mut buf), Ok(1));
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
            assert_eq!(s1.read(&mut buf), Ok(1));
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
                assert_eq!(c.read(&mut b), Ok(0));
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
            assert_eq!(s.read(&mut b), Ok(0));

            // closing should affect previous handles
            assert!(s2.write(&[0]).is_err());
            assert_eq!(s2.read(&mut b), Ok(0));

            // closing should affect new handles
            let mut s3 = t!(s.try_clone());
            assert!(s3.write(&[0]).is_err());
            assert_eq!(s3.read(&mut b), Ok(0));

            // make sure these don't die
            let _ = s2.shutdown(Shutdown::Read);
            let _ = s2.shutdown(Shutdown::Write);
            let _ = s3.shutdown(Shutdown::Read);
            let _ = s3.shutdown(Shutdown::Write);
            drop(tx);
        })
    }

    #[test]
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
            // this should wake up the child task
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

            // Enqueue a task to write to a socket
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
}
