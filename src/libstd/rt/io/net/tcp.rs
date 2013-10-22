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
use rt::io::net::ip::SocketAddr;
use rt::io::{Reader, Writer, Listener, Acceptor};
use rt::io::{io_error, read_error, EndOfFile};
use rt::rtio::{IoFactory, IoFactoryObject,
               RtioSocket,
               RtioTcpListener, RtioTcpListenerObject,
               RtioTcpAcceptor, RtioTcpAcceptorObject,
               RtioTcpStream, RtioTcpStreamObject};
use rt::local::Local;

pub struct TcpStream {
    priv obj: ~RtioTcpStreamObject
}

impl TcpStream {
    fn new(s: ~RtioTcpStreamObject) -> TcpStream {
        TcpStream { obj: s }
    }

    pub fn connect(addr: SocketAddr) -> Option<TcpStream> {
        let stream = unsafe {
            rtdebug!("borrowing io to connect");
            let io: *mut IoFactoryObject = Local::unsafe_borrow();
            rtdebug!("about to connect");
            (*io).tcp_connect(addr)
        };

        match stream {
            Ok(s) => Some(TcpStream::new(s)),
            Err(ioerr) => {
                rtdebug!("failed to connect: {:?}", ioerr);
                io_error::cond.raise(ioerr);
                None
            }
        }
    }

    pub fn peer_name(&mut self) -> Option<SocketAddr> {
        match self.obj.peer_name() {
            Ok(pn) => Some(pn),
            Err(ioerr) => {
                rtdebug!("failed to get peer name: {:?}", ioerr);
                io_error::cond.raise(ioerr);
                None
            }
        }
    }

    pub fn socket_name(&mut self) -> Option<SocketAddr> {
        match self.obj.socket_name() {
            Ok(sn) => Some(sn),
            Err(ioerr) => {
                rtdebug!("failed to get socket name: {:?}", ioerr);
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
                    read_error::cond.raise(ioerr);
                }
                return None;
            }
        }
    }

    fn eof(&mut self) -> bool { fail!() }
}

impl Writer for TcpStream {
    fn write(&mut self, buf: &[u8]) {
        match self.obj.write(buf) {
            Ok(_) => (),
            Err(ioerr) => io_error::cond.raise(ioerr),
        }
    }

    fn flush(&mut self) { /* no-op */ }
}

pub struct TcpListener {
    priv obj: ~RtioTcpListenerObject
}

impl TcpListener {
    pub fn bind(addr: SocketAddr) -> Option<TcpListener> {
        let listener = unsafe {
            let io: *mut IoFactoryObject = Local::unsafe_borrow();
            (*io).tcp_bind(addr)
        };
        match listener {
            Ok(l) => Some(TcpListener { obj: l }),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
                return None;
            }
        }
    }

    pub fn socket_name(&mut self) -> Option<SocketAddr> {
        match self.obj.socket_name() {
            Ok(sn) => Some(sn),
            Err(ioerr) => {
                rtdebug!("failed to get socket name: {:?}", ioerr);
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
    priv obj: ~RtioTcpAcceptorObject
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
    use cell::Cell;
    use rt::test::*;
    use rt::io::net::ip::{Ipv4Addr, SocketAddr};
    use rt::io::*;
    use prelude::*;
    use rt::comm::oneshot;

    #[test] #[ignore]
    fn bind_error() {
        do run_in_mt_newsched_task {
            let mut called = false;
            do io_error::cond.trap(|e| {
                assert!(e.kind == PermissionDenied);
                called = true;
            }).inside {
                let addr = SocketAddr { ip: Ipv4Addr(0, 0, 0, 0), port: 1 };
                let listener = TcpListener::bind(addr);
                assert!(listener.is_none());
            }
            assert!(called);
        }
    }

    #[test]
    fn connect_error() {
        do run_in_mt_newsched_task {
            let mut called = false;
            do io_error::cond.trap(|e| {
                let expected_error = if cfg!(unix) {
                    ConnectionRefused
                } else {
                    // On Win32, opening port 1 gives WSAEADDRNOTAVAIL error.
                    OtherIoError
                };
                assert_eq!(e.kind, expected_error);
                called = true;
            }).inside {
                let addr = SocketAddr { ip: Ipv4Addr(0, 0, 0, 0), port: 1 };
                let stream = TcpStream::connect(addr);
                assert!(stream.is_none());
            }
            assert!(called);
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
                assert!(nread.is_none());
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
                assert!(nread.is_none());
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
                let nread = stream.read(buf);
                assert!(nread.is_none());
                do read_error::cond.trap(|e| {
                    if cfg!(windows) {
                        assert_eq!(e.kind, NotConnected);
                    } else {
                        fail!();
                    }
                }).inside {
                    let nread = stream.read(buf);
                    assert!(nread.is_none());
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
                let nread = stream.read(buf);
                assert!(nread.is_none());
                do read_error::cond.trap(|e| {
                    if cfg!(windows) {
                        assert_eq!(e.kind, NotConnected);
                    } else {
                        fail!();
                    }
                }).inside {
                    let nread = stream.read(buf);
                    assert!(nread.is_none());
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
    #[ignore(cfg(windows))] // FIXME #8811
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
                    let mut stop = false;
                    do io_error::cond.trap(|e| {
                        // NB: ECONNRESET on linux, EPIPE on mac
                        assert!(e.kind == ConnectionReset || e.kind == BrokenPipe);
                        stop = true;
                    }).inside {
                        stream.write(buf);
                    }
                    if stop { break }
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
    #[ignore(cfg(windows))] // FIXME #8811
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
                    let mut stop = false;
                    do io_error::cond.trap(|e| {
                        // NB: ECONNRESET on linux, EPIPE on mac
                        assert!(e.kind == ConnectionReset || e.kind == BrokenPipe);
                        stop = true;
                    }).inside {
                        stream.write(buf);
                    }
                    if stop { break }
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
                do max.times {
                    let mut stream = TcpStream::connect(addr);
                    stream.write([99]);
                }
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
                do max.times {
                    let mut stream = TcpStream::connect(addr);
                    stream.write([99]);
                }
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
                        rtdebug!("read");
                    }
                }
            }

            port.recv();
            connect(0, addr);

            fn connect(i: int, addr: SocketAddr) {
                if i == MAX { return }

                do spawntask {
                    rtdebug!("connecting");
                    let mut stream = TcpStream::connect(addr);
                    // Connect again before writing
                    connect(i + 1, addr);
                    rtdebug!("writing");
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
                        rtdebug!("read");
                    }
                }
            }

            port.recv();
            connect(0, addr);

            fn connect(i: int, addr: SocketAddr) {
                if i == MAX { return }

                do spawntask {
                    rtdebug!("connecting");
                    let mut stream = TcpStream::connect(addr);
                    // Connect again before writing
                    connect(i + 1, addr);
                    rtdebug!("writing");
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
                        rtdebug!("read");
                    }
                }
            }

            port.recv();
            connect(0, addr);

            fn connect(i: int, addr: SocketAddr) {
                if i == MAX { return }

                do spawntask_later {
                    rtdebug!("connecting");
                    let mut stream = TcpStream::connect(addr);
                    // Connect again before writing
                    connect(i + 1, addr);
                    rtdebug!("writing");
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
                        rtdebug!("read");
                    }
                }
            }

            port.recv();
            connect(0, addr);

            fn connect(i: int, addr: SocketAddr) {
                if i == MAX { return }

                do spawntask_later {
                    rtdebug!("connecting");
                    let mut stream = TcpStream::connect(addr);
                    // Connect again before writing
                    connect(i + 1, addr);
                    rtdebug!("writing");
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
                assert!(so_name.is_some());
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

                assert!(stream.is_some());
                let mut stream = stream.unwrap();

                // Make sure peer_name gives us the
                // address/port of the peer we've
                // connected to.
                let peer_name = stream.peer_name();
                assert!(peer_name.is_some());
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
