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
use io::{Reader, Writer, Listener, Acceptor};
use io::IoResult;
use rt::rtio::{IoFactory, with_local_io,
               RtioSocket, RtioTcpListener, RtioTcpAcceptor, RtioTcpStream};

pub struct TcpStream {
    priv obj: ~RtioTcpStream
}

impl TcpStream {
    fn new(s: ~RtioTcpStream) -> TcpStream {
        TcpStream { obj: s }
    }

    pub fn connect(addr: SocketAddr) -> IoResult<TcpStream> {
        with_local_io(|io| io.tcp_connect(addr).map(TcpStream::new))
    }

    pub fn peer_name(&mut self) -> IoResult<SocketAddr> {
        self.obj.peer_name()
    }

    pub fn socket_name(&mut self) -> IoResult<SocketAddr> {
        self.obj.socket_name()
    }
}

impl Reader for TcpStream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        self.obj.read(buf)
    }
    fn eof(&mut self) -> bool { false }
}

impl Writer for TcpStream {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        self.obj.write(buf)
    }
}

pub struct TcpListener {
    priv obj: ~RtioTcpListener
}

impl TcpListener {
    pub fn bind(addr: SocketAddr) -> IoResult<TcpListener> {
        with_local_io(|io| io.tcp_bind(addr).map(|l| TcpListener { obj: l }))
    }

    pub fn socket_name(&mut self) -> IoResult<SocketAddr> {
        self.obj.socket_name()
    }
}

impl Listener<TcpStream, TcpAcceptor> for TcpListener {
    fn listen(self) -> IoResult<TcpAcceptor> {
        self.obj.listen().map(|a| TcpAcceptor { obj: a })
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
    use cell::Cell;
    use rt::test::*;
    use io::net::ip::{Ipv4Addr, SocketAddr};
    use io::*;
    use prelude::*;
    use rt::comm::oneshot;

    #[test] #[ignore]
    fn bind_error() {
        do run_in_mt_newsched_task {
            let addr = SocketAddr { ip: Ipv4Addr(0, 0, 0, 0), port: 1 };
            match TcpListener::bind(addr) {
                Ok(*) => fail!(),
                Err(e) => assert_eq!(e.kind, PermissionDenied),
            }
        }
    }

    #[test]
    fn connect_error() {
        do run_in_mt_newsched_task {
            let addr = SocketAddr { ip: Ipv4Addr(0, 0, 0, 0), port: 1 };
            match TcpStream::connect(addr) {
                Ok(*) => fail!(),
                Err(e) => assert_eq!(e.kind, if cfg!(unix) {
                    ConnectionRefused
                } else {
                    // On Win32, opening port 1 gives WSAEADDRNOTAVAIL error.
                    OtherIoError
                })
            }
        }
    }

    #[test]
    fn smoke_test_ip4() {
        do run_in_mt_newsched_task {
            let addr = next_test_ip4();
            let (port, chan) = oneshot();
            let port = Cell::new(port);
            let chan = Cell::new(chan);

            do spawntask {
                let mut acceptor = TcpListener::bind(addr).listen();
                chan.take().send(());
                let mut stream = acceptor.accept();
                let mut buf = [0];
                stream.read(buf);
                assert!(buf[0] == 99);
            }

            do spawntask {
                port.take().recv();
                let mut stream = TcpStream::connect(addr);
                stream.write([99]);
            }
        }
    }

    #[test]
    fn smoke_test_ip6() {
        do run_in_mt_newsched_task {
            let addr = next_test_ip6();
            let (port, chan) = oneshot();
            let port = Cell::new(port);
            let chan = Cell::new(chan);

            do spawntask {
                let mut acceptor = TcpListener::bind(addr).listen();
                chan.take().send(());
                let mut stream = acceptor.accept();
                let mut buf = [0];
                stream.read(buf);
                assert!(buf[0] == 99);
            }

            do spawntask {
                port.take().recv();
                let mut stream = TcpStream::connect(addr);
                stream.write([99]);
            }
        }
    }

    #[test]
    fn read_eof_ip4() {
        do run_in_mt_newsched_task {
            let addr = next_test_ip4();
            let (port, chan) = oneshot();
            let port = Cell::new(port);
            let chan = Cell::new(chan);

            do spawntask {
                let mut acceptor = TcpListener::bind(addr).listen();
                chan.take().send(());
                let mut stream = acceptor.accept();
                let mut buf = [0];
                let nread = stream.read(buf);
                assert!(nread.is_err());
            }

            do spawntask {
                port.take().recv();
                let _stream = TcpStream::connect(addr);
                // Close
            }
        }
    }

    #[test]
    fn read_eof_ip6() {
        do run_in_mt_newsched_task {
            let addr = next_test_ip6();
            let (port, chan) = oneshot();
            let port = Cell::new(port);
            let chan = Cell::new(chan);

            do spawntask {
                let mut acceptor = TcpListener::bind(addr).listen();
                chan.take().send(());
                let mut stream = acceptor.accept();
                let mut buf = [0];
                let nread = stream.read(buf);
                assert!(nread.is_err());
            }

            do spawntask {
                port.take().recv();
                let _stream = TcpStream::connect(addr);
                // Close
            }
        }
    }

    #[test]
    fn read_eof_twice_ip4() {
        do run_in_mt_newsched_task {
            let addr = next_test_ip4();
            let (port, chan) = oneshot();
            let port = Cell::new(port);
            let chan = Cell::new(chan);

            do spawntask {
                let mut acceptor = TcpListener::bind(addr).listen();
                chan.take().send(());
                let mut stream = acceptor.accept();
                let mut buf = [0];
                match stream.read(buf) {
                    Ok(*) => fail!(),
                    Err(e) => assert_eq!(e.kind, EndOfFile),
                }
                match stream.read(buf) {
                    Ok(*) => fail!(),
                    Err(e) => assert_eq!(e.kind, if cfg!(windows) {
                        NotConnected
                    } else {
                        EndOfFile
                    })
                }
            }

            do spawntask {
                port.take().recv();
                let _stream = TcpStream::connect(addr);
                // Close
            }
        }
    }

    #[test]
    fn read_eof_twice_ip6() {
        do run_in_mt_newsched_task {
            let addr = next_test_ip6();
            let (port, chan) = oneshot();
            let port = Cell::new(port);
            let chan = Cell::new(chan);

            do spawntask {
                let mut acceptor = TcpListener::bind(addr).listen();
                chan.take().send(());
                let mut stream = acceptor.accept();
                let mut buf = [0];
                match stream.read(buf) {
                    Ok(*) => fail!(),
                    Err(e) => assert_eq!(e.kind, EndOfFile),
                }
                match stream.read(buf) {
                    Ok(*) => fail!(),
                    Err(e) => assert_eq!(e.kind, if cfg!(windows) {
                        NotConnected
                    } else {
                        EndOfFile
                    })
                }
            }

            do spawntask {
                port.take().recv();
                let _stream = TcpStream::connect(addr);
                // Close
            }
        }
    }

    #[test]
    fn write_close_ip4() {
        do run_in_mt_newsched_task {
            let addr = next_test_ip4();
            let (port, chan) = oneshot();
            let port = Cell::new(port);
            let chan = Cell::new(chan);

            do spawntask {
                let mut acceptor = TcpListener::bind(addr).listen();
                chan.take().send(());
                let mut stream = acceptor.accept();
                let buf = [0];
                loop {
                    match stream.write(buf) {
                        Ok(*) => {}
                        Err(e) => {
                            // NB: ECONNRESET on linux, EPIPE on mac, ECONNABORTED
                            //     on windows
                            assert!(e.kind == ConnectionReset ||
                                    e.kind == BrokenPipe ||
                                    e.kind == ConnectionAborted,
                                    "unknown error: {:?}", e);
                            break
                        }
                    }
                }
            }

            do spawntask {
                port.take().recv();
                let _stream = TcpStream::connect(addr);
                // Close
            }
        }
    }

    #[test]
    fn write_close_ip6() {
        do run_in_mt_newsched_task {
            let addr = next_test_ip6();
            let (port, chan) = oneshot();
            let port = Cell::new(port);
            let chan = Cell::new(chan);

            do spawntask {
                let mut acceptor = TcpListener::bind(addr).listen();
                chan.take().send(());
                let mut stream = acceptor.accept();
                let buf = [0];
                loop {
                    match stream.write(buf) {
                        Ok(*) => {}
                        Err(e) => {
                            // NB: ECONNRESET on linux, EPIPE on mac, ECONNABORTED
                            //     on windows
                            assert!(e.kind == ConnectionReset ||
                                    e.kind == BrokenPipe ||
                                    e.kind == ConnectionAborted,
                                    "unknown error: {:?}", e);
                            break
                        }
                    }
                }
            }

            do spawntask {
                port.take().recv();
                let _stream = TcpStream::connect(addr);
                // Close
            }
        }
    }

    #[test]
    fn multiple_connect_serial_ip4() {
        do run_in_mt_newsched_task {
            let addr = next_test_ip4();
            let max = 10;
            let (port, chan) = oneshot();
            let port = Cell::new(port);
            let chan = Cell::new(chan);

            do spawntask {
                let mut acceptor = TcpListener::bind(addr).listen();
                chan.take().send(());
                for ref mut stream in acceptor.incoming().take(max) {
                    let mut buf = [0];
                    stream.read(buf);
                    assert_eq!(buf[0], 99);
                }
            }

            do spawntask {
                port.take().recv();
                max.times(|| {
                    let mut stream = TcpStream::connect(addr);
                    stream.write([99]);
                });
            }
        }
    }

    #[test]
    fn multiple_connect_serial_ip6() {
        do run_in_mt_newsched_task {
            let addr = next_test_ip6();
            let max = 10;
            let (port, chan) = oneshot();
            let port = Cell::new(port);
            let chan = Cell::new(chan);

            do spawntask {
                let mut acceptor = TcpListener::bind(addr).listen();
                chan.take().send(());
                for ref mut stream in acceptor.incoming().take(max) {
                    let mut buf = [0];
                    stream.read(buf);
                    assert_eq!(buf[0], 99);
                }
            }

            do spawntask {
                port.take().recv();
                max.times(|| {
                    let mut stream = TcpStream::connect(addr);
                    stream.write([99]);
                });
            }
        }
    }

    #[test]
    fn multiple_connect_interleaved_greedy_schedule_ip4() {
        do run_in_mt_newsched_task {
            let addr = next_test_ip4();
            static MAX: int = 10;
            let (port, chan) = oneshot();
            let chan = Cell::new(chan);

            do spawntask {
                let mut acceptor = TcpListener::bind(addr).listen();
                chan.take().send(());
                for (i, stream) in acceptor.incoming().enumerate().take(MAX as uint) {
                    let stream = Cell::new(stream);
                    // Start another task to handle the connection
                    do spawntask {
                        let mut stream = stream.take();
                        let mut buf = [0];
                        stream.read(buf);
                        assert!(buf[0] == i as u8);
                        debug!("read");
                    }
                }
            }

            port.recv();
            connect(0, addr);

            fn connect(i: int, addr: SocketAddr) {
                if i == MAX { return }

                do spawntask {
                    debug!("connecting");
                    let mut stream = TcpStream::connect(addr);
                    // Connect again before writing
                    connect(i + 1, addr);
                    debug!("writing");
                    stream.write([i as u8]);
                }
            }
        }
    }

    #[test]
    fn multiple_connect_interleaved_greedy_schedule_ip6() {
        do run_in_mt_newsched_task {
            let addr = next_test_ip6();
            static MAX: int = 10;
            let (port, chan) = oneshot();
            let chan = Cell::new(chan);

            do spawntask {
                let mut acceptor = TcpListener::bind(addr).listen();
                chan.take().send(());
                for (i, stream) in acceptor.incoming().enumerate().take(MAX as uint) {
                    let stream = Cell::new(stream);
                    // Start another task to handle the connection
                    do spawntask {
                        let mut stream = stream.take();
                        let mut buf = [0];
                        stream.read(buf);
                        assert!(buf[0] == i as u8);
                        debug!("read");
                    }
                }
            }

            port.recv();
            connect(0, addr);

            fn connect(i: int, addr: SocketAddr) {
                if i == MAX { return }

                do spawntask {
                    debug!("connecting");
                    let mut stream = TcpStream::connect(addr);
                    // Connect again before writing
                    connect(i + 1, addr);
                    debug!("writing");
                    stream.write([i as u8]);
                }
            }
        }
    }

    #[test]
    fn multiple_connect_interleaved_lazy_schedule_ip4() {
        do run_in_mt_newsched_task {
            let addr = next_test_ip4();
            static MAX: int = 10;
            let (port, chan) = oneshot();
            let chan = Cell::new(chan);

            do spawntask {
                let mut acceptor = TcpListener::bind(addr).listen();
                chan.take().send(());
                for stream in acceptor.incoming().take(MAX as uint) {
                    let stream = Cell::new(stream);
                    // Start another task to handle the connection
                    do spawntask_later {
                        let mut stream = stream.take();
                        let mut buf = [0];
                        stream.read(buf);
                        assert!(buf[0] == 99);
                        debug!("read");
                    }
                }
            }

            port.recv();
            connect(0, addr);

            fn connect(i: int, addr: SocketAddr) {
                if i == MAX { return }

                do spawntask_later {
                    debug!("connecting");
                    let mut stream = TcpStream::connect(addr);
                    // Connect again before writing
                    connect(i + 1, addr);
                    debug!("writing");
                    stream.write([99]);
                }
            }
        }
    }
    #[test]
    fn multiple_connect_interleaved_lazy_schedule_ip6() {
        do run_in_mt_newsched_task {
            let addr = next_test_ip6();
            static MAX: int = 10;
            let (port, chan) = oneshot();
            let chan = Cell::new(chan);

            do spawntask {
                let mut acceptor = TcpListener::bind(addr).listen();
                chan.take().send(());
                for stream in acceptor.incoming().take(MAX as uint) {
                    let stream = Cell::new(stream);
                    // Start another task to handle the connection
                    do spawntask_later {
                        let mut stream = stream.take();
                        let mut buf = [0];
                        stream.read(buf);
                        assert!(buf[0] == 99);
                        debug!("read");
                    }
                }
            }

            port.recv();
            connect(0, addr);

            fn connect(i: int, addr: SocketAddr) {
                if i == MAX { return }

                do spawntask_later {
                    debug!("connecting");
                    let mut stream = TcpStream::connect(addr);
                    // Connect again before writing
                    connect(i + 1, addr);
                    debug!("writing");
                    stream.write([99]);
                }
            }
        }
    }

    #[cfg(test)]
    fn socket_name(addr: SocketAddr) {
        do run_in_mt_newsched_task {
            do spawntask {
                let mut listener = TcpListener::bind(addr).unwrap();

                // Make sure socket_name gives
                // us the socket we binded to.
                let so_name = listener.socket_name();
                assert!(so_name.is_ok());
                assert_eq!(addr, so_name.unwrap());

            }
        }
    }

    #[cfg(test)]
    fn peer_name(addr: SocketAddr) {
        do run_in_mt_newsched_task {
            let (port, chan) = oneshot();
            let port = Cell::new(port);
            let chan = Cell::new(chan);

            do spawntask {
                let mut acceptor = TcpListener::bind(addr).listen();
                chan.take().send(());

                acceptor.accept();
            }

            do spawntask {
                port.take().recv();
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
        }
    }

    #[test]
    fn socket_and_peer_name_ip4() {
        peer_name(next_test_ip4());
        socket_name(next_test_ip4());
    }

    #[test]
    fn socket_and_peer_name_ip6() {
        // XXX: peer name is not consistent
        //peer_name(next_test_ip6());
        socket_name(next_test_ip6());
    }

}
