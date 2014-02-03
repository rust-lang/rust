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
use io::{Reader, Writer, IoResult};
use rt::rtio::{RtioSocket, RtioUdpSocket, IoFactory, LocalIo};

pub struct UdpSocket {
    priv obj: ~RtioUdpSocket
}

impl UdpSocket {
    pub fn bind(addr: SocketAddr) -> IoResult<UdpSocket> {
        LocalIo::maybe_raise(|io| {
            io.udp_bind(addr).map(|s| UdpSocket { obj: s })
        })
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
    pub fn as_socket<T>(&mut self, f: |&mut UdpSocket| -> T) -> T {
        f(&mut self.socket)
    }

    pub fn disconnect(self) -> UdpSocket { self.socket }
}

impl Reader for UdpStream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        let peer = self.connectedTo;
        self.as_socket(|sock| {
            match sock.recvfrom(buf) {
                Ok((_nread, src)) if src != peer => Ok(0),
                Ok((nread, _src)) => Ok(nread),
                Err(e) => Err(e),
            }
        })
    }
}

impl Writer for UdpStream {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        self.as_socket(|sock| sock.sendto(buf, self.connectedTo))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use io::net::ip::{SocketAddr};

    // FIXME #11530 this fails on android because tests are run as root
    iotest!(fn bind_error() {
        let addr = SocketAddr { ip: Ipv4Addr(0, 0, 0, 0), port: 1 };
        match UdpSocket::bind(addr) {
            Ok(..) => fail!(),
            Err(e) => assert_eq!(e.kind, PermissionDenied),
        }
    } #[ignore(cfg(windows))] #[ignore(cfg(target_os = "android"))])

    iotest!(fn socket_smoke_test_ip4() {
        let server_ip = next_test_ip4();
        let client_ip = next_test_ip4();
        let (port, chan) = Chan::new();
        let (port2, chan2) = Chan::new();

        spawn(proc() {
            match UdpSocket::bind(client_ip) {
                Ok(ref mut client) => {
                    port.recv();
                    client.sendto([99], server_ip).unwrap()
                }
                Err(..) => fail!()
            }
            chan2.send(());
        });

        match UdpSocket::bind(server_ip) {
            Ok(ref mut server) => {
                chan.send(());
                let mut buf = [0];
                match server.recvfrom(buf) {
                    Ok((nread, src)) => {
                        assert_eq!(nread, 1);
                        assert_eq!(buf[0], 99);
                        assert_eq!(src, client_ip);
                    }
                    Err(..) => fail!()
                }
            }
            Err(..) => fail!()
        }
        port2.recv();
    })

    iotest!(fn socket_smoke_test_ip6() {
        let server_ip = next_test_ip6();
        let client_ip = next_test_ip6();
        let (port, chan) = Chan::<()>::new();

        spawn(proc() {
            match UdpSocket::bind(client_ip) {
                Ok(ref mut client) => {
                    port.recv();
                    client.sendto([99], server_ip).unwrap()
                }
                Err(..) => fail!()
            }
        });

        match UdpSocket::bind(server_ip) {
            Ok(ref mut server) => {
                chan.send(());
                let mut buf = [0];
                match server.recvfrom(buf) {
                    Ok((nread, src)) => {
                        assert_eq!(nread, 1);
                        assert_eq!(buf[0], 99);
                        assert_eq!(src, client_ip);
                    }
                    Err(..) => fail!()
                }
            }
            Err(..) => fail!()
        }
    })

    iotest!(fn stream_smoke_test_ip4() {
        let server_ip = next_test_ip4();
        let client_ip = next_test_ip4();
        let (port, chan) = Chan::new();
        let (port2, chan2) = Chan::new();

        spawn(proc() {
            match UdpSocket::bind(client_ip) {
                Ok(client) => {
                    let client = ~client;
                    let mut stream = client.connect(server_ip);
                    port.recv();
                    stream.write([99]).unwrap();
                }
                Err(..) => fail!()
            }
            chan2.send(());
        });

        match UdpSocket::bind(server_ip) {
            Ok(server) => {
                let server = ~server;
                let mut stream = server.connect(client_ip);
                chan.send(());
                let mut buf = [0];
                match stream.read(buf) {
                    Ok(nread) => {
                        assert_eq!(nread, 1);
                        assert_eq!(buf[0], 99);
                    }
                    Err(..) => fail!()
                }
            }
            Err(..) => fail!()
        }
        port2.recv();
    })

    iotest!(fn stream_smoke_test_ip6() {
        let server_ip = next_test_ip6();
        let client_ip = next_test_ip6();
        let (port, chan) = Chan::new();
        let (port2, chan2) = Chan::new();

        spawn(proc() {
            match UdpSocket::bind(client_ip) {
                Ok(client) => {
                    let client = ~client;
                    let mut stream = client.connect(server_ip);
                    port.recv();
                    stream.write([99]).unwrap();
                }
                Err(..) => fail!()
            }
            chan2.send(());
        });

        match UdpSocket::bind(server_ip) {
            Ok(server) => {
                let server = ~server;
                let mut stream = server.connect(client_ip);
                chan.send(());
                let mut buf = [0];
                match stream.read(buf) {
                    Ok(nread) => {
                        assert_eq!(nread, 1);
                        assert_eq!(buf[0], 99);
                    }
                    Err(..) => fail!()
                }
            }
            Err(..) => fail!()
        }
        port2.recv();
    })

    pub fn socket_name(addr: SocketAddr) {
        let server = UdpSocket::bind(addr);

        assert!(server.is_ok());
        let mut server = server.unwrap();

        // Make sure socket_name gives
        // us the socket we binded to.
        let so_name = server.socket_name();
        assert!(so_name.is_ok());
        assert_eq!(addr, so_name.unwrap());
    }

    iotest!(fn socket_name_ip4() {
        socket_name(next_test_ip4());
    })

    iotest!(fn socket_name_ip6() {
        socket_name(next_test_ip6());
    })
}
