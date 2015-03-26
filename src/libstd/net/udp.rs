// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(feature = "udp", reason = "remaining functions have not been \
                                       scrutinized enough to be stabilized")]

use prelude::v1::*;

use io::{self, Error, ErrorKind};
use net::{ToSocketAddrs, SocketAddr, IpAddr};
use sys_common::net2 as net_imp;
use sys_common::AsInner;

/// A User Datagram Protocol socket.
///
/// This is an implementation of a bound UDP socket. This supports both IPv4 and
/// IPv6 addresses, and there is no corresponding notion of a server because UDP
/// is a datagram protocol.
///
/// # Examples
///
/// ```no_run
/// # #![feature(net)]
/// use std::net::UdpSocket;
///
/// # fn foo() -> std::io::Result<()> {
/// let mut socket = try!(UdpSocket::bind("127.0.0.1:34254"));
///
/// let mut buf = [0; 10];
/// let (amt, src) = try!(socket.recv_from(&mut buf));
///
/// // Send a reply to the socket we received data from
/// let buf = &mut buf[..amt];
/// buf.reverse();
/// try!(socket.send_to(buf, &src));
///
/// drop(socket); // close the socket
/// # Ok(())
/// # }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct UdpSocket(net_imp::UdpSocket);

impl UdpSocket {
    /// Creates a UDP socket from the given address.
    ///
    /// Address type can be any implementor of `ToSocketAddr` trait. See its
    /// documentation for concrete examples.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn bind<A: ToSocketAddrs>(addr: A) -> io::Result<UdpSocket> {
        super::each_addr(addr, net_imp::UdpSocket::bind).map(UdpSocket)
    }

    /// Receives data from the socket. On success, returns the number of bytes
    /// read and the address from whence the data came.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.0.recv_from(buf)
    }

    /// Sends data on the socket to the given address. Returns nothing on
    /// success.
    ///
    /// Address type can be any implementor of `ToSocketAddrs` trait. See its
    /// documentation for concrete examples.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn send_to<A: ToSocketAddrs>(&self, buf: &[u8], addr: A)
                                     -> io::Result<usize> {
        match try!(addr.to_socket_addrs()).next() {
            Some(addr) => self.0.send_to(buf, &addr),
            None => Err(Error::new(ErrorKind::InvalidInput,
                                   "no addresses to send data to", None)),
        }
    }

    /// Returns the socket address that this socket was created from.
    #[unstable(feature = "net")]
    #[deprecated(since = "1.0.0", reason = "renamed to local_addr")]
    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        self.0.socket_addr()
    }

    /// Returns the socket address that this socket was created from.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.0.socket_addr()
    }

    /// Create a new independently owned handle to the underlying socket.
    ///
    /// The returned `UdpSocket` is a reference to the same socket that this
    /// object references. Both handles will read and write the same port, and
    /// options set on one socket will be propagated to the other.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn try_clone(&self) -> io::Result<UdpSocket> {
        self.0.duplicate().map(UdpSocket)
    }

    /// Sets the broadcast flag on or off
    pub fn set_broadcast(&self, on: bool) -> io::Result<()> {
        self.0.set_broadcast(on)
    }

    /// Set the multicast loop flag to the specified value
    ///
    /// This lets multicast packets loop back to local sockets (if enabled)
    pub fn set_multicast_loop(&self, on: bool) -> io::Result<()> {
        self.0.set_multicast_loop(on)
    }

    /// Joins a multicast IP address (becomes a member of it)
    pub fn join_multicast(&self, multi: &IpAddr) -> io::Result<()> {
        self.0.join_multicast(multi)
    }

    /// Leaves a multicast IP address (drops membership from it)
    pub fn leave_multicast(&self, multi: &IpAddr) -> io::Result<()> {
        self.0.leave_multicast(multi)
    }

    /// Sets the multicast TTL
    pub fn set_multicast_time_to_live(&self, ttl: i32) -> io::Result<()> {
        self.0.multicast_time_to_live(ttl)
    }

    /// Sets this socket's TTL
    pub fn set_time_to_live(&self, ttl: i32) -> io::Result<()> {
        self.0.time_to_live(ttl)
    }
}

impl AsInner<net_imp::UdpSocket> for UdpSocket {
    fn as_inner(&self) -> &net_imp::UdpSocket { &self.0 }
}

#[cfg(test)]
mod tests {
    use prelude::v1::*;

    use io::ErrorKind;
    use net::*;
    use net::test::{next_test_ip4, next_test_ip6};
    use sync::mpsc::channel;
    use thread;

    fn each_ip(f: &mut FnMut(SocketAddr, SocketAddr)) {
        f(next_test_ip4(), next_test_ip4());
        f(next_test_ip6(), next_test_ip6());
    }

    macro_rules! t {
        ($e:expr) => {
            match $e {
                Ok(t) => t,
                Err(e) => panic!("received error for `{}`: {}", stringify!($e), e),
            }
        }
    }

    // FIXME #11530 this fails on android because tests are run as root
    #[cfg_attr(any(windows, target_os = "android"), ignore)]
    #[test]
    fn bind_error() {
        let addr = SocketAddrV4::new(Ipv4Addr::new(0, 0, 0, 0), 1);
        match UdpSocket::bind(&addr) {
            Ok(..) => panic!(),
            Err(e) => assert_eq!(e.kind(), ErrorKind::PermissionDenied),
        }
    }

    #[test]
    fn socket_smoke_test_ip4() {
        each_ip(&mut |server_ip, client_ip| {
            let (tx1, rx1) = channel();
            let (tx2, rx2) = channel();

            let _t = thread::spawn(move|| {
                let client = t!(UdpSocket::bind(&client_ip));
                rx1.recv().unwrap();
                t!(client.send_to(&[99], &server_ip));
                tx2.send(()).unwrap();
            });

            let server = t!(UdpSocket::bind(&server_ip));
            tx1.send(()).unwrap();
            let mut buf = [0];
            let (nread, src) = t!(server.recv_from(&mut buf));
            assert_eq!(nread, 1);
            assert_eq!(buf[0], 99);
            assert_eq!(src, client_ip);
            rx2.recv().unwrap();
        })
    }

    #[test]
    fn socket_name_ip4() {
        each_ip(&mut |addr, _| {
            let server = t!(UdpSocket::bind(&addr));
            assert_eq!(addr, t!(server.socket_addr()));
        })
    }

    #[test]
    fn udp_clone_smoke() {
        each_ip(&mut |addr1, addr2| {
            let sock1 = t!(UdpSocket::bind(&addr1));
            let sock2 = t!(UdpSocket::bind(&addr2));

            let _t = thread::spawn(move|| {
                let mut buf = [0, 0];
                assert_eq!(sock2.recv_from(&mut buf), Ok((1, addr1)));
                assert_eq!(buf[0], 1);
                t!(sock2.send_to(&[2], &addr1));
            });

            let sock3 = t!(sock1.try_clone());

            let (tx1, rx1) = channel();
            let (tx2, rx2) = channel();
            let _t = thread::spawn(move|| {
                rx1.recv().unwrap();
                t!(sock3.send_to(&[1], &addr2));
                tx2.send(()).unwrap();
            });
            tx1.send(()).unwrap();
            let mut buf = [0, 0];
            assert_eq!(sock1.recv_from(&mut buf), Ok((1, addr2)));
            rx2.recv().unwrap();
        })
    }

    #[test]
    fn udp_clone_two_read() {
        each_ip(&mut |addr1, addr2| {
            let sock1 = t!(UdpSocket::bind(&addr1));
            let sock2 = t!(UdpSocket::bind(&addr2));
            let (tx1, rx) = channel();
            let tx2 = tx1.clone();

            let _t = thread::spawn(move|| {
                t!(sock2.send_to(&[1], &addr1));
                rx.recv().unwrap();
                t!(sock2.send_to(&[2], &addr1));
                rx.recv().unwrap();
            });

            let sock3 = t!(sock1.try_clone());

            let (done, rx) = channel();
            let _t = thread::spawn(move|| {
                let mut buf = [0, 0];
                t!(sock3.recv_from(&mut buf));
                tx2.send(()).unwrap();
                done.send(()).unwrap();
            });
            let mut buf = [0, 0];
            t!(sock1.recv_from(&mut buf));
            tx1.send(()).unwrap();

            rx.recv().unwrap();
        })
    }

    #[test]
    fn udp_clone_two_write() {
        each_ip(&mut |addr1, addr2| {
            let sock1 = t!(UdpSocket::bind(&addr1));
            let sock2 = t!(UdpSocket::bind(&addr2));

            let (tx, rx) = channel();
            let (serv_tx, serv_rx) = channel();

            let _t = thread::spawn(move|| {
                let mut buf = [0, 1];
                rx.recv().unwrap();
                t!(sock2.recv_from(&mut buf));
                serv_tx.send(()).unwrap();
            });

            let sock3 = t!(sock1.try_clone());

            let (done, rx) = channel();
            let tx2 = tx.clone();
            let _t = thread::spawn(move|| {
                match sock3.send_to(&[1], &addr2) {
                    Ok(..) => { let _ = tx2.send(()); }
                    Err(..) => {}
                }
                done.send(()).unwrap();
            });
            match sock1.send_to(&[2], &addr2) {
                Ok(..) => { let _ = tx.send(()); }
                Err(..) => {}
            }
            drop(tx);

            rx.recv().unwrap();
            serv_rx.recv().unwrap();
        })
    }
}
