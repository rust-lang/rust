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
use rt::io::{Reader, Writer};
use rt::io::{io_error, read_error, EndOfFile};
use rt::rtio::{RtioUdpSocketObject, RtioUdpSocket, IoFactory, IoFactoryObject};
use rt::local::Local;

pub struct UdpSocket(~RtioUdpSocketObject);

impl UdpSocket {
    pub fn bind(addr: IpAddr) -> Option<UdpSocket> {
        let socket = unsafe { (*Local::unsafe_borrow::<IoFactoryObject>()).udp_bind(addr) };
        match socket {
            Ok(s) => Some(UdpSocket(s)),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
                None
            }
        }
    }

    pub fn recvfrom(&self, buf: &mut [u8]) -> Option<(uint, IpAddr)> {
        match (**self).recvfrom(buf) {
            Ok((nread, src)) => Some((nread, src)),
            Err(ioerr) => {
                // EOF is indicated by returning None
                if ioerr.kind != EndOfFile {
                    read_error::cond.raise(ioerr);
                }
                None
            }
        }
    }

    pub fn sendto(&self, buf: &[u8], dst: IpAddr) {
        match (**self).sendto(buf, dst) {
            Ok(_) => (),
            Err(ioerr) => io_error::cond.raise(ioerr),
        }
    }

    pub fn connect(self, other: IpAddr) -> UdpStream {
        UdpStream { socket: self, connectedTo: other }
    }
}

pub struct UdpStream {
    socket: UdpSocket,
    connectedTo: IpAddr
}

impl UdpStream {
    pub fn as_socket<T>(&self, f: &fn(&UdpSocket) -> T) -> T { f(&self.socket) }

    pub fn disconnect(self) -> UdpSocket { self.socket }
}

impl Reader for UdpStream {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        do self.as_socket |sock| {
            match sock.recvfrom(buf) {
                Some((_nread, src)) if src != self.connectedTo => Some(0),
                Some((nread, _src)) => Some(nread),
                None => None,
            }
        }
    }

    fn eof(&mut self) -> bool { fail!() }
}

impl Writer for UdpStream {
    fn write(&mut self, buf: &[u8]) {
        do self.as_socket |sock| {
            sock.sendto(buf, self.connectedTo);
        }
    }

    fn flush(&mut self) { fail!() }
}

#[cfg(test)]
mod test {
    use super::*;
    use rt::test::*;
    use rt::io::net::ip::Ipv4;
    use rt::io::*;
    use option::{Some, None};

    #[test]  #[ignore]
    fn bind_error() {
        do run_in_newsched_task {
            let mut called = false;
            do io_error::cond.trap(|e| {
                assert!(e.kind == PermissionDenied);
                called = true;
            }).in {
                let addr = Ipv4(0, 0, 0, 0, 1);
                let socket = UdpSocket::bind(addr);
                assert!(socket.is_none());
            }
            assert!(called);
        }
    }

    #[test]
    fn socket_smoke_test_ip4() {
        do run_in_newsched_task {
            let server_ip = next_test_ip4();
            let client_ip = next_test_ip4();

            do spawntask_immediately {
                match UdpSocket::bind(server_ip) {
                    Some(server) => {
                        let mut buf = [0];
                        match server.recvfrom(buf) {
                            Some((nread, src)) => {
                                assert_eq!(nread, 1);
                                assert_eq!(buf[0], 99);
                                assert_eq!(src, client_ip);
                            }
                            None => fail!()
                        }
                    }
                    None => fail!()
                }
            }

            do spawntask_immediately {
                match UdpSocket::bind(client_ip) {
                    Some(client) => client.sendto([99], server_ip),
                    None => fail!()
                }
            }
        }
    }

    #[test]
    fn socket_smoke_test_ip6() {
        do run_in_newsched_task {
            let server_ip = next_test_ip6();
            let client_ip = next_test_ip6();

            do spawntask_immediately {
                match UdpSocket::bind(server_ip) {
                    Some(server) => {
                        let mut buf = [0];
                        match server.recvfrom(buf) {
                            Some((nread, src)) => {
                                assert_eq!(nread, 1);
                                assert_eq!(buf[0], 99);
                                assert_eq!(src, client_ip);
                            }
                            None => fail!()
                        }
                    }
                    None => fail!()
                }
            }

            do spawntask_immediately {
                match UdpSocket::bind(client_ip) {
                    Some(client) => client.sendto([99], server_ip),
                    None => fail!()
                }
            }
        }
    }

    #[test]
    fn stream_smoke_test_ip4() {
        do run_in_newsched_task {
            let server_ip = next_test_ip4();
            let client_ip = next_test_ip4();

            do spawntask_immediately {
                match UdpSocket::bind(server_ip) {
                    Some(server) => {
                        let server = ~server;
                        let mut stream = server.connect(client_ip);
                        let mut buf = [0];
                        match stream.read(buf) {
                            Some(nread) => {
                                assert_eq!(nread, 1);
                                assert_eq!(buf[0], 99);
                            }
                            None => fail!()
                        }
                    }
                    None => fail!()
                }
            }

            do spawntask_immediately {
                match UdpSocket::bind(client_ip) {
                    Some(client) => {
                        let client = ~client;
                        let mut stream = client.connect(server_ip);
                        stream.write([99]);
                    }
                    None => fail!()
                }
            }
        }
    }

    #[test]
    fn stream_smoke_test_ip6() {
        do run_in_newsched_task {
            let server_ip = next_test_ip6();
            let client_ip = next_test_ip6();

            do spawntask_immediately {
                match UdpSocket::bind(server_ip) {
                    Some(server) => {
                        let server = ~server;
                        let mut stream = server.connect(client_ip);
                        let mut buf = [0];
                        match stream.read(buf) {
                            Some(nread) => {
                                assert_eq!(nread, 1);
                                assert_eq!(buf[0], 99);
                            }
                            None => fail!()
                        }
                    }
                    None => fail!()
                }
            }

            do spawntask_immediately {
                match UdpSocket::bind(client_ip) {
                    Some(client) => {
                        let client = ~client;
                        let mut stream = client.connect(server_ip);
                        stream.write([99]);
                    }
                    None => fail!()
                }
            }
        }
    }
}
