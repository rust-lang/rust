// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use option::{Option, Some, None};
use result::{Ok, Err};
use io::net::ip::SocketAddr;
use io::{Reader, Writer, Listener, Acceptor};
use io::{io_error, EndOfFile};
use rt::rtio::{IoFactory, LocalIo, RtioSocket, RtioTcpListener};
use rt::rtio::{RtioTcpAcceptor, RtioTcpStream};

pub struct TcpStream {
    priv obj: ~RtioTcpStream
}

impl TcpStream {
    fn new(s: ~RtioTcpStream) -> TcpStream {
        TcpStream { obj: s }
    }

    pub fn connect(addr: SocketAddr) -> Option<TcpStream> {
        LocalIo::maybe_raise(|io| {
            io.tcp_connect(addr).map(TcpStream::new)
        })
    }

    pub fn peer_name(&mut self) -> Option<SocketAddr> {
        match self.obj.peer_name() {
            Ok(pn) => Some(pn),
            Err(ioerr) => {
                debug!("failed to get peer name: {:?}", ioerr);
                io_error::cond.raise(ioerr);
                None
            }
        }
    }

    pub fn socket_name(&mut self) -> Option<SocketAddr> {
        match self.obj.socket_name() {
            Ok(sn) => Some(sn),
            Err(ioerr) => {
                debug!("failed to get socket name: {:?}", ioerr);
                io_error::cond.raise(ioerr);
                None
            }
        }
    }
}

impl Reader for TcpStream {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        match self.obj.read(buf) {
            Ok(read) => Some(read),
            Err(ioerr) => {
                // EOF is indicated by returning None
                if ioerr.kind != EndOfFile {
                    io_error::cond.raise(ioerr);
                }
                return None;
            }
        }
    }
}

impl Writer for TcpStream {
    fn write(&mut self, buf: &[u8]) {
        match self.obj.write(buf) {
            Ok(_) => (),
            Err(ioerr) => io_error::cond.raise(ioerr),
        }
    }
}

pub struct TcpListener {
    priv obj: ~RtioTcpListener
}

impl TcpListener {
    pub fn bind(addr: SocketAddr) -> Option<TcpListener> {
        LocalIo::maybe_raise(|io| {
            io.tcp_bind(addr).map(|l| TcpListener { obj: l })
        })
    }

    pub fn socket_name(&mut self) -> Option<SocketAddr> {
        match self.obj.socket_name() {
            Ok(sn) => Some(sn),
            Err(ioerr) => {
                debug!("failed to get socket name: {:?}", ioerr);
                io_error::cond.raise(ioerr);
                None
            }
        }
    }
}

impl Listener<TcpStream, TcpAcceptor> for TcpListener {
    fn listen(self) -> Option<TcpAcceptor> {
        match self.obj.listen() {
            Ok(acceptor) => Some(TcpAcceptor { obj: acceptor }),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
                None
            }
        }
    }
}

pub struct TcpAcceptor {
    priv obj: ~RtioTcpAcceptor
}

impl Acceptor<TcpStream> for TcpAcceptor {
    fn accept(&mut self) -> Option<TcpStream> {
        match self.obj.accept() {
            Ok(s) => Some(TcpStream::new(s)),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
                None
            }
        }
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
        let mut called = false;
        io_error::cond.trap(|e| {
            assert!(e.kind == PermissionDenied);
            called = true;
        }).inside(|| {
            let addr = SocketAddr { ip: Ipv4Addr(0, 0, 0, 0), port: 1 };
            let listener = TcpListener::bind(addr);
            assert!(listener.is_none());
        });
        assert!(called);
    } #[ignore(cfg(windows))] #[ignore(cfg(target_os = "android"))])

    iotest!(fn connect_error() {
        let mut called = false;
        io_error::cond.trap(|e| {
            assert_eq!(e.kind, ConnectionRefused);
            called = true;
        }).inside(|| {
            let addr = SocketAddr { ip: Ipv4Addr(0, 0, 0, 0), port: 1 };
            let stream = TcpStream::connect(addr);
            assert!(stream.is_none());
        });
        assert!(called);
    })

    iotest!(fn smoke_test_ip4() {
        let addr = next_test_ip4();
        let (port, chan) = Chan::new();

        spawn(proc() {
            port.recv();
            let mut stream = TcpStream::connect(addr);
            stream.write([99]);
        });

        let mut acceptor = TcpListener::bind(addr).listen();
        chan.send(());
        let mut stream = acceptor.accept();
        let mut buf = [0];
        stream.read(buf);
        assert!(buf[0] == 99);
    })

    iotest!(fn smoke_test_ip6() {
        let addr = next_test_ip6();
        let (port, chan) = Chan::new();

        spawn(proc() {
            port.recv();
            let mut stream = TcpStream::connect(addr);
            stream.write([99]);
        });

        let mut acceptor = TcpListener::bind(addr).listen();
        chan.send(());
        let mut stream = acceptor.accept();
        let mut buf = [0];
        stream.read(buf);
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
        assert!(nread.is_none());
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
        assert!(nread.is_none());
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
        assert!(nread.is_none());
        io_error::cond.trap(|e| {
            if cfg!(windows) {
                assert_eq!(e.kind, NotConnected);
            } else {
                fail!();
            }
        }).inside(|| {
            let nread = stream.read(buf);
            assert!(nread.is_none());
        })
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
        assert!(nread.is_none());
        io_error::cond.trap(|e| {
            if cfg!(windows) {
                assert_eq!(e.kind, NotConnected);
            } else {
                fail!();
            }
        }).inside(|| {
            let nread = stream.read(buf);
            assert!(nread.is_none());
        })
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
            let mut stop = false;
            io_error::cond.trap(|e| {
                // NB: ECONNRESET on linux, EPIPE on mac, ECONNABORTED
                //     on windows
                assert!(e.kind == ConnectionReset ||
                        e.kind == BrokenPipe ||
                        e.kind == ConnectionAborted,
                        "unknown error: {:?}", e);
                stop = true;
            }).inside(|| {
                stream.write(buf);
            });
            if stop { break }
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
            let mut stop = false;
            io_error::cond.trap(|e| {
                // NB: ECONNRESET on linux, EPIPE on mac, ECONNABORTED
                //     on windows
                assert!(e.kind == ConnectionReset ||
                        e.kind == BrokenPipe ||
                        e.kind == ConnectionAborted,
                        "unknown error: {:?}", e);
                stop = true;
            }).inside(|| {
                stream.write(buf);
            });
            if stop { break }
        }
    })

    iotest!(fn multiple_connect_serial_ip4() {
        let addr = next_test_ip4();
        let max = 10;
        let (port, chan) = Chan::new();

        spawn(proc() {
            port.recv();
            max.times(|| {
                let mut stream = TcpStream::connect(addr);
                stream.write([99]);
            });
        });

        let mut acceptor = TcpListener::bind(addr).listen();
        chan.send(());
        for ref mut stream in acceptor.incoming().take(max) {
            let mut buf = [0];
            stream.read(buf);
            assert_eq!(buf[0], 99);
        }
    })

    iotest!(fn multiple_connect_serial_ip6() {
        let addr = next_test_ip6();
        let max = 10;
        let (port, chan) = Chan::new();

        spawn(proc() {
            port.recv();
            max.times(|| {
                let mut stream = TcpStream::connect(addr);
                stream.write([99]);
            });
        });

        let mut acceptor = TcpListener::bind(addr).listen();
        chan.send(());
        for ref mut stream in acceptor.incoming().take(max) {
            let mut buf = [0];
            stream.read(buf);
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
                    stream.read(buf);
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
                stream.write([i as u8]);
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
                    stream.read(buf);
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
                stream.write([i as u8]);
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
                    stream.read(buf);
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
                stream.write([99]);
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
                    stream.read(buf);
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
                stream.write([99]);
            });
        }
    })

    pub fn socket_name(addr: SocketAddr) {
        let mut listener = TcpListener::bind(addr).unwrap();

        // Make sure socket_name gives
        // us the socket we binded to.
        let so_name = listener.socket_name();
        assert!(so_name.is_some());
        assert_eq!(addr, so_name.unwrap());
    }

    pub fn peer_name(addr: SocketAddr) {
        let (port, chan) = Chan::new();

        spawn(proc() {
            let mut acceptor = TcpListener::bind(addr).listen();
            chan.send(());
            acceptor.accept();
        });

        port.recv();
        let stream = TcpStream::connect(addr);

        assert!(stream.is_some());
        let mut stream = stream.unwrap();

        // Make sure peer_name gives us the
        // address/port of the peer we've
        // connected to.
        let peer_name = stream.peer_name();
        assert!(peer_name.is_some());
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
            let mut srv = TcpListener::bind(addr).listen();
            c.send(());
            let mut cl = srv.accept().unwrap();
            cl.write([10]);
            let mut b = [0];
            cl.read(b);
            c.send(());
        });

        p.recv();
        let mut c = TcpStream::connect(addr).unwrap();
        let mut b = [0, ..10];
        assert_eq!(c.read(b), Some(1));
        c.write([1]);
        p.recv();
    })

    iotest!(fn double_bind() {
        let mut called = false;
        io_error::cond.trap(|e| {
            assert!(e.kind == ConnectionRefused || e.kind == OtherIoError);
            called = true;
        }).inside(|| {
            let addr = next_test_ip4();
            let listener = TcpListener::bind(addr).unwrap().listen();
            assert!(listener.is_some());
            let listener2 = TcpListener::bind(addr).and_then(|l|
                                                    l.listen());
            assert!(listener2.is_none());
        });
        assert!(called);
    })

    iotest!(fn fast_rebind() {
        let addr = next_test_ip4();
        let (port, chan) = Chan::new();

        spawn(proc() {
            port.recv();
            let stream = TcpStream::connect(addr);
            // Close
            port.recv();
        });

        {
            let mut acceptor = TcpListener::bind(addr).listen();
            chan.send(());
            {
                let stream = acceptor.accept();
                // Close client
                chan.send(());
            }
            // Close listener
        }
        let listener = TcpListener::bind(addr);
    })
}
