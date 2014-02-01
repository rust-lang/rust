// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io::net::ip::SocketAddr;
use io::{Reader, Writer, Listener, Acceptor, IoResult};
use rt::rtio::{IoFactory, LocalIo, RtioSocket, RtioTcpListener};
use rt::rtio::{RtioTcpAcceptor, RtioTcpStream};

pub struct TcpStream {
    priv obj: ~RtioTcpStream
}

impl TcpStream {
    fn new(s: ~RtioTcpStream) -> TcpStream {
        TcpStream { obj: s }
    }

    pub fn connect(addr: SocketAddr) -> IoResult<TcpStream> {
        LocalIo::maybe_raise(|io| {
            io.tcp_connect(addr).map(TcpStream::new)
        })
    }

    pub fn peer_name(&mut self) -> IoResult<SocketAddr> {
        self.obj.peer_name()
    }

    pub fn socket_name(&mut self) -> IoResult<SocketAddr> {
        self.obj.socket_name()
    }
}

impl Reader for TcpStream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { self.obj.read(buf) }
}

impl Writer for TcpStream {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { self.obj.write(buf) }
}

pub struct TcpListener {
    priv obj: ~RtioTcpListener
}

impl TcpListener {
    pub fn bind(addr: SocketAddr) -> IoResult<TcpListener> {
        LocalIo::maybe_raise(|io| {
            io.tcp_bind(addr).map(|l| TcpListener { obj: l })
        })
    }

    pub fn socket_name(&mut self) -> IoResult<SocketAddr> {
        self.obj.socket_name()
    }
}

impl Listener<TcpStream, TcpAcceptor> for TcpListener {
    fn listen(self) -> IoResult<TcpAcceptor> {
        self.obj.listen().map(|acceptor| TcpAcceptor { obj: acceptor })
    }
}

pub struct TcpAcceptor {
    priv obj: ~RtioTcpAcceptor
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
        let (port, chan) = Chan::new();

        spawn(proc() {
            port.recv();
            let mut stream = TcpStream::connect(addr);
            stream.write([99]).unwrap();
        });

        let mut acceptor = TcpListener::bind(addr).listen();
        chan.send(());
        let mut stream = acceptor.accept();
        let mut buf = [0];
        stream.read(buf).unwrap();
        assert!(buf[0] == 99);
    })

    iotest!(fn smoke_test_ip6() {
        let addr = next_test_ip6();
        let (port, chan) = Chan::new();

        spawn(proc() {
            port.recv();
            let mut stream = TcpStream::connect(addr);
            stream.write([99]).unwrap();
        });

        let mut acceptor = TcpListener::bind(addr).listen();
        chan.send(());
        let mut stream = acceptor.accept();
        let mut buf = [0];
        stream.read(buf).unwrap();
        assert!(buf[0] == 99);
    })

    iotest!(fn read_eof_ip4() {
        let addr = next_test_ip4();
        let (port, chan) = Chan::new();

        spawn(proc() {
            port.recv();
            let _stream = TcpStream::connect(addr);
            // Close
        });

        let mut acceptor = TcpListener::bind(addr).listen();
        chan.send(());
        let mut stream = acceptor.accept();
        let mut buf = [0];
        let nread = stream.read(buf);
        assert!(nread.is_err());
    })

    iotest!(fn read_eof_ip6() {
        let addr = next_test_ip6();
        let (port, chan) = Chan::new();

        spawn(proc() {
            port.recv();
            let _stream = TcpStream::connect(addr);
            // Close
        });

        let mut acceptor = TcpListener::bind(addr).listen();
        chan.send(());
        let mut stream = acceptor.accept();
        let mut buf = [0];
        let nread = stream.read(buf);
        assert!(nread.is_err());
    })

    iotest!(fn read_eof_twice_ip4() {
        let addr = next_test_ip4();
        let (port, chan) = Chan::new();

        spawn(proc() {
            port.recv();
            let _stream = TcpStream::connect(addr);
            // Close
        });

        let mut acceptor = TcpListener::bind(addr).listen();
        chan.send(());
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
        let (port, chan) = Chan::new();

        spawn(proc() {
            port.recv();
            let _stream = TcpStream::connect(addr);
            // Close
        });

        let mut acceptor = TcpListener::bind(addr).listen();
        chan.send(());
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
        let (port, chan) = Chan::new();

        spawn(proc() {
            port.recv();
            let _stream = TcpStream::connect(addr);
            // Close
        });

        let mut acceptor = TcpListener::bind(addr).listen();
        chan.send(());
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
        let (port, chan) = Chan::new();

        spawn(proc() {
            port.recv();
            let _stream = TcpStream::connect(addr);
            // Close
        });

        let mut acceptor = TcpListener::bind(addr).listen();
        chan.send(());
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
        let (port, chan) = Chan::new();

        spawn(proc() {
            port.recv();
            for _ in range(0, max) {
                let mut stream = TcpStream::connect(addr);
                stream.write([99]).unwrap();
            }
        });

        let mut acceptor = TcpListener::bind(addr).listen();
        chan.send(());
        for ref mut stream in acceptor.incoming().take(max) {
            let mut buf = [0];
            stream.read(buf).unwrap();
            assert_eq!(buf[0], 99);
        }
    })

    iotest!(fn multiple_connect_serial_ip6() {
        let addr = next_test_ip6();
        let max = 10u;
        let (port, chan) = Chan::new();

        spawn(proc() {
            port.recv();
            for _ in range(0, max) {
                let mut stream = TcpStream::connect(addr);
                stream.write([99]).unwrap();
            }
        });

        let mut acceptor = TcpListener::bind(addr).listen();
        chan.send(());
        for ref mut stream in acceptor.incoming().take(max) {
            let mut buf = [0];
            stream.read(buf).unwrap();
            assert_eq!(buf[0], 99);
        }
    })

    iotest!(fn multiple_connect_interleaved_greedy_schedule_ip4() {
        let addr = next_test_ip4();
        static MAX: int = 10;
        let (port, chan) = Chan::new();

        spawn(proc() {
            let mut acceptor = TcpListener::bind(addr).listen();
            chan.send(());
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

        port.recv();
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
        let (port, chan) = Chan::<()>::new();

        spawn(proc() {
            let mut acceptor = TcpListener::bind(addr).listen();
            chan.send(());
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

        port.recv();
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
        let addr = next_test_ip4();
        static MAX: int = 10;
        let (port, chan) = Chan::new();

        spawn(proc() {
            let mut acceptor = TcpListener::bind(addr).listen();
            chan.send(());
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

        port.recv();
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
        let addr = next_test_ip6();
        static MAX: int = 10;
        let (port, chan) = Chan::new();

        spawn(proc() {
            let mut acceptor = TcpListener::bind(addr).listen();
            chan.send(());
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

        port.recv();
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
        let (port, chan) = Chan::new();

        spawn(proc() {
            let mut acceptor = TcpListener::bind(addr).listen();
            chan.send(());
            acceptor.accept().unwrap();
        });

        port.recv();
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
        let (p, c) = Chan::new();
        spawn(proc() {
            let mut srv = TcpListener::bind(addr).listen().unwrap();
            c.send(());
            let mut cl = srv.accept().unwrap();
            cl.write([10]).unwrap();
            let mut b = [0];
            cl.read(b).unwrap();
            c.send(());
        });

        p.recv();
        let mut c = TcpStream::connect(addr).unwrap();
        let mut b = [0, ..10];
        assert_eq!(c.read(b), Ok(1));
        c.write([1]).unwrap();
        p.recv();
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
        let (port, chan) = Chan::new();

        spawn(proc() {
            port.recv();
            let _stream = TcpStream::connect(addr).unwrap();
            // Close
            port.recv();
        });

        {
            let mut acceptor = TcpListener::bind(addr).listen();
            chan.send(());
            {
                let _stream = acceptor.accept().unwrap();
                // Close client
                chan.send(());
            }
            // Close listener
        }
        let _listener = TcpListener::bind(addr);
    })
}
