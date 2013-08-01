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
use rt::io::net::ip::IpAddr;
use rt::io::{Reader, Writer, Listener};
use rt::io::{io_error, read_error, EndOfFile};
use rt::rtio::{IoFactory, IoFactoryObject,
               RtioSocket, RtioTcpListener,
               RtioTcpListenerObject, RtioTcpStream,
               RtioTcpStreamObject};
use rt::local::Local;

pub struct TcpStream(~RtioTcpStreamObject);

impl TcpStream {
    fn new(s: ~RtioTcpStreamObject) -> TcpStream {
        TcpStream(s)
    }

    pub fn connect(addr: IpAddr) -> Option<TcpStream> {
        let stream = unsafe {
            rtdebug!("borrowing io to connect");
            let io = Local::unsafe_borrow::<IoFactoryObject>();
            rtdebug!("about to connect");
            (*io).tcp_connect(addr)
        };

        match stream {
            Ok(s) => Some(TcpStream::new(s)),
            Err(ioerr) => {
                rtdebug!("failed to connect: %?", ioerr);
                io_error::cond.raise(ioerr);
                None
            }
        }
    }

    pub fn peer_name(&mut self) -> Option<IpAddr> {
        match (**self).peer_name() {
            Ok(pn) => Some(pn),
            Err(ioerr) => {
                rtdebug!("failed to get peer name: %?", ioerr);
                io_error::cond.raise(ioerr);
                None
            }
        }
    }

    pub fn socket_name(&mut self) -> Option<IpAddr> {
        match (**self).socket_name() {
            Ok(sn) => Some(sn),
            Err(ioerr) => {
                rtdebug!("failed to get socket name: %?", ioerr);
                io_error::cond.raise(ioerr);
                None
            }
        }
    }
}

impl Reader for TcpStream {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        match (**self).read(buf) {
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
        match (**self).write(buf) {
            Ok(_) => (),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
            }
        }
    }

    fn flush(&mut self) { fail!() }
}

pub struct TcpListener(~RtioTcpListenerObject);

impl TcpListener {
    pub fn bind(addr: IpAddr) -> Option<TcpListener> {
        let listener = unsafe {
            let io = Local::unsafe_borrow::<IoFactoryObject>();
            (*io).tcp_bind(addr)
        };
        match listener {
            Ok(l) => Some(TcpListener(l)),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
                return None;
            }
        }
    }

    pub fn socket_name(&mut self) -> Option<IpAddr> {
        match (**self).socket_name() {
            Ok(sn) => Some(sn),
            Err(ioerr) => {
                rtdebug!("failed to get socket name: %?", ioerr);
                io_error::cond.raise(ioerr);
                None
            }
        }
    }
}

impl Listener<TcpStream> for TcpListener {
    fn accept(&mut self) -> Option<TcpStream> {
        match (**self).accept() {
            Ok(s) => {
                Some(TcpStream::new(s))
            }
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
                return None;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use cell::Cell;
    use rt::test::*;
    use rt::io::net::ip::Ipv4;
    use rt::io::*;
    use prelude::*;

    #[test] #[ignore]
    fn bind_error() {
        do run_in_newsched_task {
            let mut called = false;
            do io_error::cond.trap(|e| {
                assert!(e.kind == PermissionDenied);
                called = true;
            }).inside {
                let addr = Ipv4(0, 0, 0, 0, 1);
                let listener = TcpListener::bind(addr);
                assert!(listener.is_none());
            }
            assert!(called);
        }
    }

    #[test]
    fn connect_error() {
        do run_in_newsched_task {
            let mut called = false;
            do io_error::cond.trap(|e| {
                assert!(e.kind == ConnectionRefused);
                called = true;
            }).inside {
                let addr = Ipv4(0, 0, 0, 0, 1);
                let stream = TcpStream::connect(addr);
                assert!(stream.is_none());
            }
            assert!(called);
        }
    }

    #[test]
    fn smoke_test_ip4() {
        do run_in_newsched_task {
            let addr = next_test_ip4();

            do spawntask {
                let mut listener = TcpListener::bind(addr);
                let mut stream = listener.accept();
                let mut buf = [0];
                stream.read(buf);
                assert!(buf[0] == 99);
            }

            do spawntask {
                let mut stream = TcpStream::connect(addr);
                stream.write([99]);
            }
        }
    }

    #[test]
    fn smoke_test_ip6() {
        do run_in_newsched_task {
            let addr = next_test_ip6();

            do spawntask {
                let mut listener = TcpListener::bind(addr);
                let mut stream = listener.accept();
                let mut buf = [0];
                stream.read(buf);
                assert!(buf[0] == 99);
            }

            do spawntask {
                let mut stream = TcpStream::connect(addr);
                stream.write([99]);
            }
        }
    }

    #[test]
    fn read_eof_ip4() {
        do run_in_newsched_task {
            let addr = next_test_ip4();

            do spawntask {
                let mut listener = TcpListener::bind(addr);
                let mut stream = listener.accept();
                let mut buf = [0];
                let nread = stream.read(buf);
                assert!(nread.is_none());
            }

            do spawntask {
                let _stream = TcpStream::connect(addr);
                // Close
            }
        }
    }

    #[test]
    fn read_eof_ip6() {
        do run_in_newsched_task {
            let addr = next_test_ip6();

            do spawntask {
                let mut listener = TcpListener::bind(addr);
                let mut stream = listener.accept();
                let mut buf = [0];
                let nread = stream.read(buf);
                assert!(nread.is_none());
            }

            do spawntask {
                let _stream = TcpStream::connect(addr);
                // Close
            }
        }
    }

    #[test]
    fn read_eof_twice_ip4() {
        do run_in_newsched_task {
            let addr = next_test_ip4();

            do spawntask {
                let mut listener = TcpListener::bind(addr);
                let mut stream = listener.accept();
                let mut buf = [0];
                let nread = stream.read(buf);
                assert!(nread.is_none());
                let nread = stream.read(buf);
                assert!(nread.is_none());
            }

            do spawntask {
                let _stream = TcpStream::connect(addr);
                // Close
            }
        }
    }

    #[test]
    fn read_eof_twice_ip6() {
        do run_in_newsched_task {
            let addr = next_test_ip6();

            do spawntask {
                let mut listener = TcpListener::bind(addr);
                let mut stream = listener.accept();
                let mut buf = [0];
                let nread = stream.read(buf);
                assert!(nread.is_none());
                let nread = stream.read(buf);
                assert!(nread.is_none());
            }

            do spawntask {
                let _stream = TcpStream::connect(addr);
                // Close
            }
        }
    }

    #[test]
    fn write_close_ip4() {
        do run_in_newsched_task {
            let addr = next_test_ip4();

            do spawntask {
                let mut listener = TcpListener::bind(addr);
                let mut stream = listener.accept();
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
                let _stream = TcpStream::connect(addr);
                // Close
            }
        }
    }

    #[test]
    fn write_close_ip6() {
        do run_in_newsched_task {
            let addr = next_test_ip6();

            do spawntask {
                let mut listener = TcpListener::bind(addr);
                let mut stream = listener.accept();
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
                let _stream = TcpStream::connect(addr);
                // Close
            }
        }
    }

    #[test]
    fn multiple_connect_serial_ip4() {
        do run_in_newsched_task {
            let addr = next_test_ip4();
            let max = 10;

            do spawntask {
                let mut listener = TcpListener::bind(addr);
                do max.times {
                    let mut stream = listener.accept();
                    let mut buf = [0];
                    stream.read(buf);
                    assert_eq!(buf[0], 99);
                }
            }

            do spawntask {
                do max.times {
                    let mut stream = TcpStream::connect(addr);
                    stream.write([99]);
                }
            }
        }
    }

    #[test]
    fn multiple_connect_serial_ip6() {
        do run_in_newsched_task {
            let addr = next_test_ip6();
            let max = 10;

            do spawntask {
                let mut listener = TcpListener::bind(addr);
                do max.times {
                    let mut stream = listener.accept();
                    let mut buf = [0];
                    stream.read(buf);
                    assert_eq!(buf[0], 99);
                }
            }

            do spawntask {
                do max.times {
                    let mut stream = TcpStream::connect(addr);
                    stream.write([99]);
                }
            }
        }
    }

    #[test]
    fn multiple_connect_interleaved_greedy_schedule_ip4() {
        do run_in_newsched_task {
            let addr = next_test_ip4();
            static MAX: int = 10;

            do spawntask {
                let mut listener = TcpListener::bind(addr);
                foreach i in range(0, MAX) {
                    let stream = Cell::new(listener.accept());
                    rtdebug!("accepted");
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

            connect(0, addr);

            fn connect(i: int, addr: IpAddr) {
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
        do run_in_newsched_task {
            let addr = next_test_ip6();
            static MAX: int = 10;

            do spawntask {
                let mut listener = TcpListener::bind(addr);
                foreach i in range(0, MAX) {
                    let stream = Cell::new(listener.accept());
                    rtdebug!("accepted");
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

            connect(0, addr);

            fn connect(i: int, addr: IpAddr) {
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
        do run_in_newsched_task {
            let addr = next_test_ip4();
            static MAX: int = 10;

            do spawntask {
                let mut listener = TcpListener::bind(addr);
                foreach _ in range(0, MAX) {
                    let stream = Cell::new(listener.accept());
                    rtdebug!("accepted");
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

            connect(0, addr);

            fn connect(i: int, addr: IpAddr) {
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
        do run_in_newsched_task {
            let addr = next_test_ip6();
            static MAX: int = 10;

            do spawntask {
                let mut listener = TcpListener::bind(addr);
                foreach _ in range(0, MAX) {
                    let stream = Cell::new(listener.accept());
                    rtdebug!("accepted");
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

            connect(0, addr);

            fn connect(i: int, addr: IpAddr) {
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
    fn socket_name(addr: IpAddr) {
        do run_in_newsched_task {
            do spawntask {
                let listener = TcpListener::bind(addr);

                assert!(listener.is_some());
                let mut listener = listener.unwrap();

                // Make sure socket_name gives
                // us the socket we binded to.
                let so_name = listener.socket_name();
                assert!(so_name.is_some());
                assert_eq!(addr, so_name.unwrap());

            }
        }
    }

    #[cfg(test)]
    fn peer_name(addr: IpAddr) {
        do run_in_newsched_task {
            do spawntask {
                let mut listener = TcpListener::bind(addr);

                listener.accept();
            }

            do spawntask {
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
