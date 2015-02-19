// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! UDP (User Datagram Protocol) network connections.
//!
//! This module contains the ability to open a UDP stream to a socket address.
//! The destination and binding addresses can either be an IPv4 or IPv6
//! address. There is no corresponding notion of a server because UDP is a
//! datagram protocol.

use clone::Clone;
use old_io::net::ip::{SocketAddr, IpAddr, ToSocketAddr};
use old_io::IoResult;
use option::Option;
use sys::udp::UdpSocket as UdpSocketImp;
use sys_common;

/// A User Datagram Protocol socket.
///
/// This is an implementation of a bound UDP socket. This supports both IPv4 and
/// IPv6 addresses, and there is no corresponding notion of a server because UDP
/// is a datagram protocol.
///
/// # Example
///
/// ```rust,no_run
/// # #![allow(unused_must_use)]
///
/// use std::old_io::net::udp::UdpSocket;
/// use std::old_io::net::ip::{Ipv4Addr, SocketAddr};
/// fn main() {
///     let addr = SocketAddr { ip: Ipv4Addr(127, 0, 0, 1), port: 34254 };
///     let mut socket = match UdpSocket::bind(addr) {
///         Ok(s) => s,
///         Err(e) => panic!("couldn't bind socket: {}", e),
///     };
///
///     let mut buf = [0; 10];
///     match socket.recv_from(&mut buf) {
///         Ok((amt, src)) => {
///             // Send a reply to the socket we received data from
///             let buf = &mut buf[..amt];
///             buf.reverse();
///             socket.send_to(buf, src);
///         }
///         Err(e) => println!("couldn't receive a datagram: {}", e)
///     }
///     drop(socket); // close the socket
/// }
/// ```
pub struct UdpSocket {
    inner: UdpSocketImp,
}

impl UdpSocket {
    /// Creates a UDP socket from the given address.
    ///
    /// Address type can be any implementor of `ToSocketAddr` trait. See its
    /// documentation for concrete examples.
    pub fn bind<A: ToSocketAddr>(addr: A) -> IoResult<UdpSocket> {
        super::with_addresses(addr, |addr| {
            UdpSocketImp::bind(addr).map(|s| UdpSocket { inner: s })
        })
    }

    /// Receives data from the socket. On success, returns the number of bytes
    /// read and the address from whence the data came.
    pub fn recv_from(&mut self, buf: &mut [u8]) -> IoResult<(uint, SocketAddr)> {
        self.inner.recv_from(buf)
    }

    /// Sends data on the socket to the given address. Returns nothing on
    /// success.
    ///
    /// Address type can be any implementer of `ToSocketAddr` trait. See its
    /// documentation for concrete examples.
    pub fn send_to<A: ToSocketAddr>(&mut self, buf: &[u8], addr: A) -> IoResult<()> {
        super::with_addresses(addr, |addr| self.inner.send_to(buf, addr))
    }

    /// Returns the socket address that this socket was created from.
    pub fn socket_name(&mut self) -> IoResult<SocketAddr> {
        self.inner.socket_name()
    }

    /// Joins a multicast IP address (becomes a member of it)
    #[unstable(feature = "io")]
    pub fn join_multicast(&mut self, multi: IpAddr) -> IoResult<()> {
        self.inner.join_multicast(multi)
    }

    /// Leaves a multicast IP address (drops membership from it)
    #[unstable(feature = "io")]
    pub fn leave_multicast(&mut self, multi: IpAddr) -> IoResult<()> {
        self.inner.leave_multicast(multi)
    }

    /// Set the multicast loop flag to the specified value
    ///
    /// This lets multicast packets loop back to local sockets (if enabled)
    #[unstable(feature = "io")]
    pub fn set_multicast_loop(&mut self, on: bool) -> IoResult<()> {
        self.inner.set_multicast_loop(on)
    }

    /// Sets the multicast TTL
    #[unstable(feature = "io")]
    pub fn set_multicast_ttl(&mut self, ttl: int) -> IoResult<()> {
        self.inner.multicast_time_to_live(ttl)
    }

    /// Sets this socket's TTL
    #[unstable(feature = "io")]
    pub fn set_ttl(&mut self, ttl: int) -> IoResult<()> {
        self.inner.time_to_live(ttl)
    }

    /// Sets the broadcast flag on or off
    #[unstable(feature = "io")]
    pub fn set_broadcast(&mut self, broadcast: bool) -> IoResult<()> {
        self.inner.set_broadcast(broadcast)
    }

    /// Sets the read/write timeout for this socket.
    ///
    /// For more information, see `TcpStream::set_timeout`
    #[unstable(feature = "io",
               reason = "the timeout argument may change in type and value")]
    pub fn set_timeout(&mut self, timeout_ms: Option<u64>) {
        self.inner.set_timeout(timeout_ms)
    }

    /// Sets the read timeout for this socket.
    ///
    /// For more information, see `TcpStream::set_timeout`
    #[unstable(feature = "io",
               reason = "the timeout argument may change in type and value")]
    pub fn set_read_timeout(&mut self, timeout_ms: Option<u64>) {
        self.inner.set_read_timeout(timeout_ms)
    }

    /// Sets the write timeout for this socket.
    ///
    /// For more information, see `TcpStream::set_timeout`
    #[unstable(feature = "io",
               reason = "the timeout argument may change in type and value")]
    pub fn set_write_timeout(&mut self, timeout_ms: Option<u64>) {
        self.inner.set_write_timeout(timeout_ms)
    }
}

impl Clone for UdpSocket {
    /// Creates a new handle to this UDP socket, allowing for simultaneous
    /// reads and writes of the socket.
    ///
    /// The underlying UDP socket will not be closed until all handles to the
    /// socket have been deallocated. Two concurrent reads will not receive
    /// the same data. Instead, the first read will receive the first packet
    /// received, and the second read will receive the second packet.
    fn clone(&self) -> UdpSocket {
        UdpSocket {
            inner: self.inner.clone(),
        }
    }
}

impl sys_common::AsInner<UdpSocketImp> for UdpSocket {
    fn as_inner(&self) -> &UdpSocketImp {
        &self.inner
    }
}

#[cfg(test)]
mod test {
    use prelude::v1::*;

    use sync::mpsc::channel;
    use old_io::net::ip::*;
    use old_io::test::*;
    use old_io::{IoError, TimedOut, PermissionDenied, ShortWrite};
    use super::*;
    use thread;

    // FIXME #11530 this fails on android because tests are run as root
    #[cfg_attr(any(windows, target_os = "android"), ignore)]
    #[test]
    fn bind_error() {
        let addr = SocketAddr { ip: Ipv4Addr(0, 0, 0, 0), port: 1 };
        match UdpSocket::bind(addr) {
            Ok(..) => panic!(),
            Err(e) => assert_eq!(e.kind, PermissionDenied),
        }
    }

    #[test]
    fn socket_smoke_test_ip4() {
        let server_ip = next_test_ip4();
        let client_ip = next_test_ip4();
        let (tx1, rx1) = channel();
        let (tx2, rx2) = channel();

        let _t = thread::spawn(move|| {
            match UdpSocket::bind(client_ip) {
                Ok(ref mut client) => {
                    rx1.recv().unwrap();
                    client.send_to(&[99], server_ip).unwrap()
                }
                Err(..) => panic!()
            }
            tx2.send(()).unwrap();
        });

        match UdpSocket::bind(server_ip) {
            Ok(ref mut server) => {
                tx1.send(()).unwrap();
                let mut buf = [0];
                match server.recv_from(&mut buf) {
                    Ok((nread, src)) => {
                        assert_eq!(nread, 1);
                        assert_eq!(buf[0], 99);
                        assert_eq!(src, client_ip);
                    }
                    Err(..) => panic!()
                }
            }
            Err(..) => panic!()
        }
        rx2.recv().unwrap();
    }

    #[test]
    fn socket_smoke_test_ip6() {
        let server_ip = next_test_ip6();
        let client_ip = next_test_ip6();
        let (tx, rx) = channel::<()>();

        let _t = thread::spawn(move|| {
            match UdpSocket::bind(client_ip) {
                Ok(ref mut client) => {
                    rx.recv().unwrap();
                    client.send_to(&[99], server_ip).unwrap()
                }
                Err(..) => panic!()
            }
        });

        match UdpSocket::bind(server_ip) {
            Ok(ref mut server) => {
                tx.send(()).unwrap();
                let mut buf = [0];
                match server.recv_from(&mut buf) {
                    Ok((nread, src)) => {
                        assert_eq!(nread, 1);
                        assert_eq!(buf[0], 99);
                        assert_eq!(src, client_ip);
                    }
                    Err(..) => panic!()
                }
            }
            Err(..) => panic!()
        }
    }

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

    #[test]
    fn socket_name_ip4() {
        socket_name(next_test_ip4());
    }

    #[test]
    fn socket_name_ip6() {
        socket_name(next_test_ip6());
    }

    #[test]
    fn udp_clone_smoke() {
        let addr1 = next_test_ip4();
        let addr2 = next_test_ip4();
        let mut sock1 = UdpSocket::bind(addr1).unwrap();
        let sock2 = UdpSocket::bind(addr2).unwrap();

        let _t = thread::spawn(move|| {
            let mut sock2 = sock2;
            let mut buf = [0, 0];
            assert_eq!(sock2.recv_from(&mut buf), Ok((1, addr1)));
            assert_eq!(buf[0], 1);
            sock2.send_to(&[2], addr1).unwrap();
        });

        let sock3 = sock1.clone();

        let (tx1, rx1) = channel();
        let (tx2, rx2) = channel();
        let _t = thread::spawn(move|| {
            let mut sock3 = sock3;
            rx1.recv().unwrap();
            sock3.send_to(&[1], addr2).unwrap();
            tx2.send(()).unwrap();
        });
        tx1.send(()).unwrap();
        let mut buf = [0, 0];
        assert_eq!(sock1.recv_from(&mut buf), Ok((1, addr2)));
        rx2.recv().unwrap();
    }

    #[test]
    fn udp_clone_two_read() {
        let addr1 = next_test_ip4();
        let addr2 = next_test_ip4();
        let mut sock1 = UdpSocket::bind(addr1).unwrap();
        let sock2 = UdpSocket::bind(addr2).unwrap();
        let (tx1, rx) = channel();
        let tx2 = tx1.clone();

        let _t = thread::spawn(move|| {
            let mut sock2 = sock2;
            sock2.send_to(&[1], addr1).unwrap();
            rx.recv().unwrap();
            sock2.send_to(&[2], addr1).unwrap();
            rx.recv().unwrap();
        });

        let sock3 = sock1.clone();

        let (done, rx) = channel();
        let _t = thread::spawn(move|| {
            let mut sock3 = sock3;
            let mut buf = [0, 0];
            sock3.recv_from(&mut buf).unwrap();
            tx2.send(()).unwrap();
            done.send(()).unwrap();
        });
        let mut buf = [0, 0];
        sock1.recv_from(&mut buf).unwrap();
        tx1.send(()).unwrap();

        rx.recv().unwrap();
    }

    #[test]
    fn udp_clone_two_write() {
        let addr1 = next_test_ip4();
        let addr2 = next_test_ip4();
        let mut sock1 = UdpSocket::bind(addr1).unwrap();
        let sock2 = UdpSocket::bind(addr2).unwrap();

        let (tx, rx) = channel();
        let (serv_tx, serv_rx) = channel();

        let _t = thread::spawn(move|| {
            let mut sock2 = sock2;
            let mut buf = [0, 1];

            rx.recv().unwrap();
            match sock2.recv_from(&mut buf) {
                Ok(..) => {}
                Err(e) => panic!("failed receive: {}", e),
            }
            serv_tx.send(()).unwrap();
        });

        let sock3 = sock1.clone();

        let (done, rx) = channel();
        let tx2 = tx.clone();
        let _t = thread::spawn(move|| {
            let mut sock3 = sock3;
            match sock3.send_to(&[1], addr2) {
                Ok(..) => { let _ = tx2.send(()); }
                Err(..) => {}
            }
            done.send(()).unwrap();
        });
        match sock1.send_to(&[2], addr2) {
            Ok(..) => { let _ = tx.send(()); }
            Err(..) => {}
        }
        drop(tx);

        rx.recv().unwrap();
        serv_rx.recv().unwrap();
    }

    #[cfg(not(windows))] // FIXME #17553
    #[test]
    fn recv_from_timeout() {
        let addr1 = next_test_ip4();
        let addr2 = next_test_ip4();
        let mut a = UdpSocket::bind(addr1).unwrap();
        let a2 = UdpSocket::bind(addr2).unwrap();

        let (tx, rx) = channel();
        let (tx2, rx2) = channel();
        let _t = thread::spawn(move|| {
            let mut a = a2;
            assert_eq!(a.recv_from(&mut [0]), Ok((1, addr1)));
            assert_eq!(a.send_to(&[0], addr1), Ok(()));
            rx.recv().unwrap();
            assert_eq!(a.send_to(&[0], addr1), Ok(()));

            tx2.send(()).unwrap();
        });

        // Make sure that reads time out, but writes can continue
        a.set_read_timeout(Some(20));
        assert_eq!(a.recv_from(&mut [0]).err().unwrap().kind, TimedOut);
        assert_eq!(a.recv_from(&mut [0]).err().unwrap().kind, TimedOut);
        assert_eq!(a.send_to(&[0], addr2), Ok(()));

        // Cloned handles should be able to block
        let mut a2 = a.clone();
        assert_eq!(a2.recv_from(&mut [0]), Ok((1, addr2)));

        // Clearing the timeout should allow for receiving
        a.set_timeout(None);
        tx.send(()).unwrap();
        assert_eq!(a2.recv_from(&mut [0]), Ok((1, addr2)));

        // Make sure the child didn't die
        rx2.recv().unwrap();
    }

    #[test]
    fn send_to_timeout() {
        let addr1 = next_test_ip4();
        let addr2 = next_test_ip4();
        let mut a = UdpSocket::bind(addr1).unwrap();
        let _b = UdpSocket::bind(addr2).unwrap();

        a.set_write_timeout(Some(1000));
        for _ in 0..100 {
            match a.send_to(&[0;4*1024], addr2) {
                Ok(()) | Err(IoError { kind: ShortWrite(..), .. }) => {},
                Err(IoError { kind: TimedOut, .. }) => break,
                Err(e) => panic!("other error: {}", e),
            }
        }
    }
}
