// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
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
use io::net::ip::SocketAddr;
use io::{Reader, Writer, Listener, Acceptor};
use kinds::Send;
use rt::rtio::{IoFactory, LocalIo, RtioSocket, RtioTcpListener};
use rt::rtio::{RtioTcpAcceptor, RtioTcpStream};

/// A structure which represents a TCP stream between a local socket and a
/// remote socket.
///
/// # Example
///
/// ```rust
/// # #[allow(unused_must_use)];
/// use std::io::net::tcp::TcpStream;
/// use std::io::net::ip::{Ipv4Addr, SocketAddr};
///
/// let addr = SocketAddr { ip: Ipv4Addr(127, 0, 0, 1), port: 34254 };
/// let mut stream = TcpStream::connect(addr);
///
/// stream.write([1]);
/// let mut buf = [0];
/// stream.read(buf);
/// drop(stream); // close the connection
/// ```
pub struct TcpStream {
    obj: ~RtioTcpStream:Send
}

impl TcpStream {
    fn new(s: ~RtioTcpStream:Send) -> TcpStream {
        TcpStream { obj: s }
    }

    /// Creates a TCP connection to a remote socket address.
    ///
    /// If no error is encountered, then `Ok(stream)` is returned.
    pub fn connect(addr: SocketAddr) -> IoResult<TcpStream> {
        LocalIo::maybe_raise(|io| {
            io.tcp_connect(addr).map(TcpStream::new)
        })
    }

    /// Returns the socket address of the remote peer of this TCP connection.
    pub fn peer_name(&mut self) -> IoResult<SocketAddr> {
        self.obj.peer_name()
    }

    /// Returns the socket address of the local half of this TCP connection.
    pub fn socket_name(&mut self) -> IoResult<SocketAddr> {
        self.obj.socket_name()
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
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { self.obj.read(buf) }
}

impl Writer for TcpStream {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { self.obj.write(buf) }
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
/// use std::io::net::ip::{Ipv4Addr, SocketAddr};
/// use std::io::{Acceptor, Listener};
///
/// let addr = SocketAddr { ip: Ipv4Addr(127, 0, 0, 1), port: 80 };
/// let listener = TcpListener::bind(addr);
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
    obj: ~RtioTcpListener:Send
}

impl TcpListener {
    /// Creates a new `TcpListener` which will be bound to the specified local
    /// socket address. This listener is not ready for accepting connections,
    /// `listen` must be called on it before that's possible.
    ///
    /// Binding with a port number of 0 will request that the OS assigns a port
    /// to this listener. The port allocated can be queried via the
    /// `socket_name` function.
    pub fn bind(addr: SocketAddr) -> IoResult<TcpListener> {
        LocalIo::maybe_raise(|io| {
            io.tcp_bind(addr).map(|l| TcpListener { obj: l })
        })
    }

    /// Returns the local socket address of this listener.
    pub fn socket_name(&mut self) -> IoResult<SocketAddr> {
        self.obj.socket_name()
    }
}

impl Listener<TcpStream, TcpAcceptor> for TcpListener {
    fn listen(self) -> IoResult<TcpAcceptor> {
        self.obj.listen().map(|acceptor| TcpAcceptor { obj: acceptor })
    }
}

/// The accepting half of a TCP socket server. This structure is created through
/// a `TcpListener`'s `listen` method, and this object can be used to accept new
/// `TcpStream` instances.
pub struct TcpAcceptor {
    obj: ~RtioTcpAcceptor:Send
}

impl Acceptor<TcpStream> for TcpAcceptor {
    fn accept(&mut self) -> IoResult<TcpStream> {
        self.obj.accept().map(TcpStream::new)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use io::net::ip::SocketAddr;
    use io::*;
    use prelude::*;

    // FIXME #11530 this fails on android because tests are run as root
    iotest!(fn bind_error() {
        let addr = SocketAddr { ip: Ipv4Addr(0, 0, 0, 0), port: 1 };
        match TcpListener::bind(addr) {
            Ok(..) => fail!(),
            Err(e) => assert_eq!(e.kind, PermissionDenied),
        }
    } #[ignore(cfg(windows))] #[ignore(cfg(target_os = "android"))])

    iotest!(fn connect_error() {
        let addr = SocketAddr { ip: Ipv4Addr(0, 0, 0, 0), port: 1 };
        match TcpStream::connect(addr) {
            Ok(..) => fail!(),
            Err(e) => assert_eq!(e.kind, ConnectionRefused),
        }
    })

    iotest!(fn smoke_test_ip4() {
        let addr = next_test_ip4();
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let mut stream = TcpStream::connect(addr);
            stream.write([99]).unwrap();
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        stream.read(buf).unwrap();
        assert!(buf[0] == 99);
    })

    iotest!(fn smoke_test_ip6() {
        let addr = next_test_ip6();
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let mut stream = TcpStream::connect(addr);
            stream.write([99]).unwrap();
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        stream.read(buf).unwrap();
        assert!(buf[0] == 99);
    })

    iotest!(fn read_eof_ip4() {
        let addr = next_test_ip4();
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let _stream = TcpStream::connect(addr);
            // Close
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        let nread = stream.read(buf);
        assert!(nread.is_err());
    })

    iotest!(fn read_eof_ip6() {
        let addr = next_test_ip6();
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let _stream = TcpStream::connect(addr);
            // Close
        });

        let mut stream = acceptor.accept();
        let mut buf = [0];
        let nread = stream.read(buf);
        assert!(nread.is_err());
    })

    iotest!(fn read_eof_twice_ip4() {
        let addr = next_test_ip4();
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let _stream = TcpStream::connect(addr);
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
                        "unknown kind: {:?}", e.kind);
            }
        }
    })

    iotest!(fn read_eof_twice_ip6() {
        let addr = next_test_ip6();
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let _stream = TcpStream::connect(addr);
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
                        "unknown kind: {:?}", e.kind);
            }
        }
    })

    iotest!(fn write_close_ip4() {
        let addr = next_test_ip4();
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let _stream = TcpStream::connect(addr);
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
                            "unknown error: {:?}", e);
                    break;
                }
            }
        }
    })

    iotest!(fn write_close_ip6() {
        let addr = next_test_ip6();
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let _stream = TcpStream::connect(addr);
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
                            "unknown error: {:?}", e);
                    break;
                }
            }
        }
    })

    iotest!(fn multiple_connect_serial_ip4() {
        let addr = next_test_ip4();
        let max = 10u;
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            for _ in range(0, max) {
                let mut stream = TcpStream::connect(addr);
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
        let max = 10u;
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            for _ in range(0, max) {
                let mut stream = TcpStream::connect(addr);
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
        static MAX: int = 10;
        let acceptor = TcpListener::bind(addr).listen();

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
            if i == MAX { return }

            spawn(proc() {
                debug!("connecting");
                let mut stream = TcpStream::connect(addr);
                // Connect again before writing
                connect(i + 1, addr);
                debug!("writing");
                stream.write([i as u8]).unwrap();
            });
        }
    })

    iotest!(fn multiple_connect_interleaved_greedy_schedule_ip6() {
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
                    stream.read(buf).unwrap();
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
                stream.write([i as u8]).unwrap();
            });
        }
    })

    iotest!(fn multiple_connect_interleaved_lazy_schedule_ip4() {
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
                    stream.read(buf).unwrap();
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
                stream.write([99]).unwrap();
            });
        }
    })

    iotest!(fn multiple_connect_interleaved_lazy_schedule_ip6() {
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
                    stream.read(buf).unwrap();
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
                stream.write([99]).unwrap();
            });
        }
    })

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
        let (tx, rx) = channel();
        spawn(proc() {
            let mut srv = TcpListener::bind(addr).listen().unwrap();
            tx.send(());
            let mut cl = srv.accept().unwrap();
            cl.write([10]).unwrap();
            let mut b = [0];
            cl.read(b).unwrap();
            tx.send(());
        });

        rx.recv();
        let mut c = TcpStream::connect(addr).unwrap();
        let mut b = [0, ..10];
        assert_eq!(c.read(b), Ok(1));
        c.write([1]).unwrap();
        rx.recv();
    })

    iotest!(fn double_bind() {
        let addr = next_test_ip4();
        let listener = TcpListener::bind(addr).unwrap().listen();
        assert!(listener.is_ok());
        match TcpListener::bind(addr).listen() {
            Ok(..) => fail!(),
            Err(e) => {
                assert!(e.kind == ConnectionRefused || e.kind == OtherIoError);
            }
        }
    })

    iotest!(fn fast_rebind() {
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
    })

    iotest!(fn tcp_clone_smoke() {
        let addr = next_test_ip4();
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let mut s = TcpStream::connect(addr);
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
        let mut acceptor = TcpListener::bind(addr).listen();
        let (tx1, rx) = channel();
        let tx2 = tx1.clone();

        spawn(proc() {
            let mut s = TcpStream::connect(addr);
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
        let mut acceptor = TcpListener::bind(addr).listen();

        spawn(proc() {
            let mut s = TcpStream::connect(addr);
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
        let a = TcpListener::bind(addr).unwrap().listen();
        spawn(proc() {
            let mut a = a;
            let mut c = a.accept().unwrap();
            assert_eq!(c.read_to_end(), Ok(vec!()));
            c.write([1]).unwrap();
        });

        let mut s = TcpStream::connect(addr).unwrap();
        assert!(s.obj.close_write().is_ok());
        assert!(s.write([1]).is_err());
        assert_eq!(s.read_to_end(), Ok(vec!(1)));
    })
}
