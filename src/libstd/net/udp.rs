// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use fmt;
use io::{self, Error, ErrorKind};
use net::{ToSocketAddrs, SocketAddr};
use sys_common::net as net_imp;
use sys_common::{AsInner, FromInner, IntoInner};
use time::Duration;

/// A User Datagram Protocol socket.
///
/// This is an implementation of a bound UDP socket. This supports both IPv4 and
/// IPv6 addresses, and there is no corresponding notion of a server because UDP
/// is a datagram protocol.
///
/// # Examples
///
/// ```no_run
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
    /// The address type can be any implementor of `ToSocketAddr` trait. See
    /// its documentation for concrete examples.
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

    /// Sends data on the socket to the given address. On success, returns the
    /// number of bytes written.
    ///
    /// Address type can be any implementor of `ToSocketAddrs` trait. See its
    /// documentation for concrete examples.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn send_to<A: ToSocketAddrs>(&self, buf: &[u8], addr: A)
                                     -> io::Result<usize> {
        match try!(addr.to_socket_addrs()).next() {
            Some(addr) => self.0.send_to(buf, &addr),
            None => Err(Error::new(ErrorKind::InvalidInput,
                                   "no addresses to send data to")),
        }
    }

    /// Returns the socket address that this socket was created from.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.0.socket_addr()
    }

    /// Creates a new independently owned handle to the underlying socket.
    ///
    /// The returned `UdpSocket` is a reference to the same socket that this
    /// object references. Both handles will read and write the same port, and
    /// options set on one socket will be propagated to the other.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn try_clone(&self) -> io::Result<UdpSocket> {
        self.0.duplicate().map(UdpSocket)
    }

    /// Sets the read timeout to the timeout specified.
    ///
    /// If the value specified is `None`, then `read` calls will block
    /// indefinitely. It is an error to pass the zero `Duration` to this
    /// method.
    ///
    /// # Note
    ///
    /// Platforms may return a different error code whenever a read times out as
    /// a result of setting this option. For example Unix typically returns an
    /// error of the kind `WouldBlock`, but Windows may return `TimedOut`.
    #[stable(feature = "socket_timeout", since = "1.4.0")]
    pub fn set_read_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        self.0.set_read_timeout(dur)
    }

    /// Sets the write timeout to the timeout specified.
    ///
    /// If the value specified is `None`, then `write` calls will block
    /// indefinitely. It is an error to pass the zero `Duration` to this
    /// method.
    ///
    /// # Note
    ///
    /// Platforms may return a different error code whenever a write times out
    /// as a result of setting this option. For example Unix typically returns
    /// an error of the kind `WouldBlock`, but Windows may return `TimedOut`.
    #[stable(feature = "socket_timeout", since = "1.4.0")]
    pub fn set_write_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        self.0.set_write_timeout(dur)
    }

    /// Returns the read timeout of this socket.
    ///
    /// If the timeout is `None`, then `read` calls will block indefinitely.
    #[stable(feature = "socket_timeout", since = "1.4.0")]
    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        self.0.read_timeout()
    }

    /// Returns the write timeout of this socket.
    ///
    /// If the timeout is `None`, then `write` calls will block indefinitely.
    #[stable(feature = "socket_timeout", since = "1.4.0")]
    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        self.0.write_timeout()
    }
}

impl AsInner<net_imp::UdpSocket> for UdpSocket {
    fn as_inner(&self) -> &net_imp::UdpSocket { &self.0 }
}

impl FromInner<net_imp::UdpSocket> for UdpSocket {
    fn from_inner(inner: net_imp::UdpSocket) -> UdpSocket { UdpSocket(inner) }
}

impl IntoInner<net_imp::UdpSocket> for UdpSocket {
    fn into_inner(self) -> net_imp::UdpSocket { self.0 }
}

impl fmt::Debug for UdpSocket {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use prelude::v1::*;

    use io::ErrorKind;
    use net::*;
    use net::test::{next_test_ip4, next_test_ip6};
    use sync::mpsc::channel;
    use sys_common::AsInner;
    use time::Duration;
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

    #[test]
    fn bind_error() {
        match UdpSocket::bind("1.1.1.1:9999") {
            Ok(..) => panic!(),
            Err(e) => {
                assert_eq!(e.kind(), ErrorKind::AddrNotAvailable)
            }
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
            assert_eq!(addr, t!(server.local_addr()));
        })
    }

    #[test]
    fn udp_clone_smoke() {
        each_ip(&mut |addr1, addr2| {
            let sock1 = t!(UdpSocket::bind(&addr1));
            let sock2 = t!(UdpSocket::bind(&addr2));

            let _t = thread::spawn(move|| {
                let mut buf = [0, 0];
                assert_eq!(sock2.recv_from(&mut buf).unwrap(), (1, addr1));
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
            assert_eq!(sock1.recv_from(&mut buf).unwrap(), (1, addr2));
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

    #[test]
    fn debug() {
        let name = if cfg!(windows) {"socket"} else {"fd"};
        let socket_addr = next_test_ip4();

        let udpsock = t!(UdpSocket::bind(&socket_addr));
        let udpsock_inner = udpsock.0.socket().as_inner();
        let compare = format!("UdpSocket {{ addr: {:?}, {}: {:?} }}",
                              socket_addr, name, udpsock_inner);
        assert_eq!(format!("{:?}", udpsock), compare);
    }

    // FIXME: re-enabled bitrig/openbsd/netbsd tests once their socket timeout code
    //        no longer has rounding errors.
    #[cfg_attr(any(target_os = "bitrig", target_os = "netbsd", target_os = "openbsd"), ignore)]
    #[test]
    fn timeouts() {
        let addr = next_test_ip4();

        let stream = t!(UdpSocket::bind(&addr));
        let dur = Duration::new(15410, 0);

        assert_eq!(None, t!(stream.read_timeout()));

        t!(stream.set_read_timeout(Some(dur)));
        assert_eq!(Some(dur), t!(stream.read_timeout()));

        assert_eq!(None, t!(stream.write_timeout()));

        t!(stream.set_write_timeout(Some(dur)));
        assert_eq!(Some(dur), t!(stream.write_timeout()));

        t!(stream.set_read_timeout(None));
        assert_eq!(None, t!(stream.read_timeout()));

        t!(stream.set_write_timeout(None));
        assert_eq!(None, t!(stream.write_timeout()));
    }

    #[test]
    fn test_read_timeout() {
        let addr = next_test_ip4();

        let mut stream = t!(UdpSocket::bind(&addr));
        t!(stream.set_read_timeout(Some(Duration::from_millis(1000))));

        let mut buf = [0; 10];
        let wait = Duration::span(|| {
            let kind = stream.recv_from(&mut buf).err().expect("expected error").kind();
            assert!(kind == ErrorKind::WouldBlock || kind == ErrorKind::TimedOut);
        });
        assert!(wait > Duration::from_millis(400));
    }

    #[test]
    fn test_read_with_timeout() {
        let addr = next_test_ip4();

        let mut stream = t!(UdpSocket::bind(&addr));
        t!(stream.set_read_timeout(Some(Duration::from_millis(1000))));

        t!(stream.send_to(b"hello world", &addr));

        let mut buf = [0; 11];
        t!(stream.recv_from(&mut buf));
        assert_eq!(b"hello world", &buf[..]);

        let wait = Duration::span(|| {
            let kind = stream.recv_from(&mut buf).err().expect("expected error").kind();
            assert!(kind == ErrorKind::WouldBlock || kind == ErrorKind::TimedOut);
        });
        assert!(wait > Duration::from_millis(400));
    }
}
