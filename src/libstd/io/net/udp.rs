// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use result::{Ok, Err};
use io::net::ip::SocketAddr;
use io::{Reader, Writer};
use io::IoResult;
use rt::rtio::{RtioSocket, RtioUdpSocket, IoFactory, with_local_io};

pub struct UdpSocket {
    priv obj: ~RtioUdpSocket
}

impl UdpSocket {
    pub fn bind(addr: SocketAddr) -> IoResult<UdpSocket> {
        with_local_io(|io| io.udp_bind(addr).map(|s| UdpSocket { obj: s }))
    }

    pub fn recvfrom(&mut self, buf: &mut [u8]) -> IoResult<(uint, SocketAddr)> {
        self.obj.recvfrom(buf)
    }

    pub fn sendto(&mut self, buf: &[u8], dst: SocketAddr) -> IoResult<()> {
        self.obj.sendto(buf, dst)
    }

    pub fn connect(self, other: SocketAddr) -> UdpStream {
        UdpStream { socket: self, connectedTo: other }
    }

    pub fn socket_name(&mut self) -> IoResult<SocketAddr> {
        self.obj.socket_name()
    }
}

pub struct UdpStream {
    priv socket: UdpSocket,
    priv connectedTo: SocketAddr
}

impl UdpStream {
    pub fn as_socket<'a>(&'a mut self) -> &'a mut UdpSocket {
        &mut self.socket
    }

    pub fn disconnect(self) -> UdpSocket { self.socket }
}

impl Reader for UdpStream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        let peer = self.connectedTo;
        let sock = self.as_socket();
        match if_ok!(sock.recvfrom(buf)) {
            (_nread, src) if src != peer => Ok(0),
            (nread, _src) => Ok(nread),
        }
    }

    fn eof(&mut self) -> bool { false }
}

impl Writer for UdpStream {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        self.as_socket().sendto(buf, self.connectedTo)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rt::test::*;
    use io::net::ip::{Ipv4Addr, SocketAddr};
    use io::*;
    use result::{Ok, Err};
    use rt::comm::oneshot;
    use cell::Cell;

    #[test]  #[ignore]
    fn bind_error() {
        do run_in_mt_newsched_task {
            let addr = SocketAddr { ip: Ipv4Addr(0, 0, 0, 0), port: 1 };
            match UdpSocket::bind(addr) {
                Ok(*) => fail!(),
                Err(e) => assert_eq!(e.kind, PermissionDenied),
            }
        }
    }

    #[test]
    fn socket_smoke_test_ip4() {
        do run_in_mt_newsched_task {
            let server_ip = next_test_ip4();
            let client_ip = next_test_ip4();
            let (port, chan) = oneshot();
            let port = Cell::new(port);
            let chan = Cell::new(chan);

            do spawntask {
                match UdpSocket::bind(server_ip) {
                    Ok(ref mut server) => {
                        chan.take().send(());
                        let mut buf = [0];
                        match server.recvfrom(buf) {
                            Ok((nread, src)) => {
                                assert_eq!(nread, 1);
                                assert_eq!(buf[0], 99);
                                assert_eq!(src, client_ip);
                            }
                            Err(*) => fail!()
                        }
                    }
                    Err(*) => fail!()
                }
            }

            do spawntask {
                match UdpSocket::bind(client_ip) {
                    Ok(ref mut client) => {
                        port.take().recv();
                        client.sendto([99], server_ip);
                    }
                    Err(*) => fail!()
                }
            }
        }
    }

    #[test]
    fn socket_smoke_test_ip6() {
        do run_in_mt_newsched_task {
            let server_ip = next_test_ip6();
            let client_ip = next_test_ip6();
            let (port, chan) = oneshot();
            let port = Cell::new(port);
            let chan = Cell::new(chan);

            do spawntask {
                match UdpSocket::bind(server_ip) {
                    Ok(ref mut server) => {
                        chan.take().send(());
                        let mut buf = [0];
                        match server.recvfrom(buf) {
                            Ok((nread, src)) => {
                                assert_eq!(nread, 1);
                                assert_eq!(buf[0], 99);
                                assert_eq!(src, client_ip);
                            }
                            Err(*)  => fail!()
                        }
                    }
                    Err(*) => fail!()
                }
            }

            do spawntask {
                match UdpSocket::bind(client_ip) {
                    Ok(ref mut client) => {
                        port.take().recv();
                        client.sendto([99], server_ip);
                    }
                    Err(*) => fail!()
                }
            }
        }
    }

    #[test]
    fn stream_smoke_test_ip4() {
        do run_in_mt_newsched_task {
            let server_ip = next_test_ip4();
            let client_ip = next_test_ip4();
            let (port, chan) = oneshot();
            let port = Cell::new(port);
            let chan = Cell::new(chan);

            do spawntask {
                match UdpSocket::bind(server_ip) {
                    Ok(server) => {
                        let server = ~server;
                        let mut stream = server.connect(client_ip);
                        chan.take().send(());
                        let mut buf = [0];
                        match stream.read(buf) {
                            Ok(nread) => {
                                assert_eq!(nread, 1);
                                assert_eq!(buf[0], 99);
                            }
                            Err(*) => fail!()
                        }
                    }
                    Err(*) => fail!()
                }
            }

            do spawntask {
                match UdpSocket::bind(client_ip) {
                    Ok(client) => {
                        let client = ~client;
                        let mut stream = client.connect(server_ip);
                        port.take().recv();
                        stream.write([99]);
                    }
                    Err(*) => fail!()
                }
            }
        }
    }

    #[test]
    fn stream_smoke_test_ip6() {
        do run_in_mt_newsched_task {
            let server_ip = next_test_ip6();
            let client_ip = next_test_ip6();
            let (port, chan) = oneshot();
            let port = Cell::new(port);
            let chan = Cell::new(chan);

            do spawntask {
                match UdpSocket::bind(server_ip) {
                    Ok(server) => {
                        let server = ~server;
                        let mut stream = server.connect(client_ip);
                        chan.take().send(());
                        let mut buf = [0];
                        match stream.read(buf) {
                            Ok(nread) => {
                                assert_eq!(nread, 1);
                                assert_eq!(buf[0], 99);
                            }
                            Err(*) => fail!()
                        }
                    }
                    Err(*) => fail!()
                }
            }

            do spawntask {
                match UdpSocket::bind(client_ip) {
                    Ok(client) => {
                        let client = ~client;
                        let mut stream = client.connect(server_ip);
                        port.take().recv();
                        stream.write([99]);
                    }
                    Err(*) => fail!()
                }
            }
        }
    }

    #[cfg(test)]
    fn socket_name(addr: SocketAddr) {
        do run_in_mt_newsched_task {
            do spawntask {
                let server = UdpSocket::bind(addr);

                assert!(server.is_ok());
                let mut server = server.unwrap();

                // Make sure socket_name gives
                // us the socket we binded to.
                let so_name = server.socket_name();
                assert!(so_name.is_ok());
                assert_eq!(addr, so_name.unwrap());

            }
        }
    }

    #[test]
    fn socket_name_ip4() {
        socket_name(next_test_ip4());
    }

    #[test]
    fn socket_name_ip6() {
        socket_name(next_test_ip6());
    }
}
