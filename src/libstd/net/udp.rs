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
use net::{ToSocketAddrs, SocketAddr, Ipv4Addr, Ipv6Addr};
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
/// {
///     let mut socket = UdpSocket::bind("127.0.0.1:34254")?;
///
///     // read from the socket
///     let mut buf = [0; 10];
///     let (amt, src) = socket.recv_from(&mut buf)?;
///
///     // send a reply to the socket we received data from
///     let buf = &mut buf[..amt];
///     buf.reverse();
///     socket.send_to(buf, &src)?;
///     # Ok(())
/// } // the socket is closed here
/// # }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct UdpSocket(net_imp::UdpSocket);

impl UdpSocket {
    /// Creates a UDP socket from the given address.
    ///
    /// The address type can be any implementor of [`ToSocketAddrs`] trait. See
    /// its documentation for concrete examples.
    ///
    /// [`ToSocketAddrs`]: ../../std/net/trait.ToSocketAddrs.html
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn bind<A: ToSocketAddrs>(addr: A) -> io::Result<UdpSocket> {
        super::each_addr(addr, net_imp::UdpSocket::bind).map(UdpSocket)
    }

    /// Receives data from the socket. On success, returns the number of bytes
    /// read and the address from whence the data came.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// let mut buf = [0; 10];
    /// let (number_of_bytes, src_addr) = socket.recv_from(&mut buf)
    ///                                         .expect("Didn't receive data");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.0.recv_from(buf)
    }

    /// Sends data on the socket to the given address. On success, returns the
    /// number of bytes written.
    ///
    /// Address type can be any implementor of [`ToSocketAddrs`] trait. See its
    /// documentation for concrete examples.
    ///
    /// This will return an error when the IP version of the local socket
    /// does not match that returned from [`ToSocketAddrs`].
    ///
    /// See https://github.com/rust-lang/rust/issues/34202 for more details.
    ///
    /// [`ToSocketAddrs`]: ../../std/net/trait.ToSocketAddrs.html
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// socket.send_to(&[0; 10], "127.0.0.1:4242").expect("couldn't send data");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn send_to<A: ToSocketAddrs>(&self, buf: &[u8], addr: A)
                                     -> io::Result<usize> {
        match addr.to_socket_addrs()?.next() {
            Some(addr) => self.0.send_to(buf, &addr),
            None => Err(Error::new(ErrorKind::InvalidInput,
                                   "no addresses to send data to")),
        }
    }

    /// Returns the socket address that this socket was created from.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, UdpSocket};
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// assert_eq!(socket.local_addr().unwrap(),
    ///            SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 34254)));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.0.socket_addr()
    }

    /// Creates a new independently owned handle to the underlying socket.
    ///
    /// The returned `UdpSocket` is a reference to the same socket that this
    /// object references. Both handles will read and write the same port, and
    /// options set on one socket will be propagated to the other.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// let socket_clone = socket.try_clone().expect("couldn't clone the socket");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn try_clone(&self) -> io::Result<UdpSocket> {
        self.0.duplicate().map(UdpSocket)
    }

    /// Sets the read timeout to the timeout specified.
    ///
    /// If the value specified is [`None`], then [`read()`] calls will block
    /// indefinitely. It is an error to pass the zero [`Duration`] to this
    /// method.
    ///
    /// # Note
    ///
    /// Platforms may return a different error code whenever a read times out as
    /// a result of setting this option. For example Unix typically returns an
    /// error of the kind [`WouldBlock`], but Windows may return [`TimedOut`].
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    /// [`read()`]: ../../std/io/trait.Read.html#tymethod.read
    /// [`Duration`]: ../../std/time/struct.Duration.html
    /// [`WouldBlock`]: ../../std/io/enum.ErrorKind.html#variant.WouldBlock
    /// [`TimedOut`]: ../../std/io/enum.ErrorKind.html#variant.TimedOut
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// socket.set_read_timeout(None).expect("set_read_timeout call failed");
    /// ```
    #[stable(feature = "socket_timeout", since = "1.4.0")]
    pub fn set_read_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        self.0.set_read_timeout(dur)
    }

    /// Sets the write timeout to the timeout specified.
    ///
    /// If the value specified is [`None`], then [`write()`] calls will block
    /// indefinitely. It is an error to pass the zero [`Duration`] to this
    /// method.
    ///
    /// # Note
    ///
    /// Platforms may return a different error code whenever a write times out
    /// as a result of setting this option. For example Unix typically returns
    /// an error of the kind [`WouldBlock`], but Windows may return [`TimedOut`].
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    /// [`write()`]: ../../std/io/trait.Write.html#tymethod.write
    /// [`Duration`]: ../../std/time/struct.Duration.html
    /// [`WouldBlock`]: ../../std/io/enum.ErrorKind.html#variant.WouldBlock
    /// [`TimedOut`]: ../../std/io/enum.ErrorKind.html#variant.TimedOut
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// socket.set_write_timeout(None).expect("set_write_timeout call failed");
    /// ```
    #[stable(feature = "socket_timeout", since = "1.4.0")]
    pub fn set_write_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        self.0.set_write_timeout(dur)
    }

    /// Returns the read timeout of this socket.
    ///
    /// If the timeout is [`None`], then [`read()`] calls will block indefinitely.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    /// [`read()`]: ../../std/io/trait.Read.html#tymethod.read
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// socket.set_read_timeout(None).expect("set_read_timeout call failed");
    /// assert_eq!(socket.read_timeout().unwrap(), None);
    /// ```
    #[stable(feature = "socket_timeout", since = "1.4.0")]
    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        self.0.read_timeout()
    }

    /// Returns the write timeout of this socket.
    ///
    /// If the timeout is [`None`], then [`write()`] calls will block indefinitely.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    /// [`write()`]: ../../std/io/trait.Write.html#tymethod.write
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// socket.set_write_timeout(None).expect("set_write_timeout call failed");
    /// assert_eq!(socket.write_timeout().unwrap(), None);
    /// ```
    #[stable(feature = "socket_timeout", since = "1.4.0")]
    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        self.0.write_timeout()
    }

    /// Sets the value of the `SO_BROADCAST` option for this socket.
    ///
    /// When enabled, this socket is allowed to send packets to a broadcast
    /// address.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// socket.set_broadcast(false).expect("set_broadcast call failed");
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn set_broadcast(&self, broadcast: bool) -> io::Result<()> {
        self.0.set_broadcast(broadcast)
    }

    /// Gets the value of the `SO_BROADCAST` option for this socket.
    ///
    /// For more information about this option, see
    /// [`set_broadcast`][link].
    ///
    /// [link]: #method.set_broadcast
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// socket.set_broadcast(false).expect("set_broadcast call failed");
    /// assert_eq!(socket.broadcast().unwrap(), false);
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn broadcast(&self) -> io::Result<bool> {
        self.0.broadcast()
    }

    /// Sets the value of the `IP_MULTICAST_LOOP` option for this socket.
    ///
    /// If enabled, multicast packets will be looped back to the local socket.
    /// Note that this may not have any affect on IPv6 sockets.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// socket.set_multicast_loop_v4(false).expect("set_multicast_loop_v4 call failed");
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn set_multicast_loop_v4(&self, multicast_loop_v4: bool) -> io::Result<()> {
        self.0.set_multicast_loop_v4(multicast_loop_v4)
    }

    /// Gets the value of the `IP_MULTICAST_LOOP` option for this socket.
    ///
    /// For more information about this option, see
    /// [`set_multicast_loop_v4`][link].
    ///
    /// [link]: #method.set_multicast_loop_v4
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// socket.set_multicast_loop_v4(false).expect("set_multicast_loop_v4 call failed");
    /// assert_eq!(socket.multicast_loop_v4().unwrap(), false);
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn multicast_loop_v4(&self) -> io::Result<bool> {
        self.0.multicast_loop_v4()
    }

    /// Sets the value of the `IP_MULTICAST_TTL` option for this socket.
    ///
    /// Indicates the time-to-live value of outgoing multicast packets for
    /// this socket. The default value is 1 which means that multicast packets
    /// don't leave the local network unless explicitly requested.
    ///
    /// Note that this may not have any affect on IPv6 sockets.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// socket.set_multicast_ttl_v4(42).expect("set_multicast_ttl_v4 call failed");
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn set_multicast_ttl_v4(&self, multicast_ttl_v4: u32) -> io::Result<()> {
        self.0.set_multicast_ttl_v4(multicast_ttl_v4)
    }

    /// Gets the value of the `IP_MULTICAST_TTL` option for this socket.
    ///
    /// For more information about this option, see
    /// [`set_multicast_ttl_v4`][link].
    ///
    /// [link]: #method.set_multicast_ttl_v4
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// socket.set_multicast_ttl_v4(42).expect("set_multicast_ttl_v4 call failed");
    /// assert_eq!(socket.multicast_ttl_v4().unwrap(), 42);
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn multicast_ttl_v4(&self) -> io::Result<u32> {
        self.0.multicast_ttl_v4()
    }

    /// Sets the value of the `IPV6_MULTICAST_LOOP` option for this socket.
    ///
    /// Controls whether this socket sees the multicast packets it sends itself.
    /// Note that this may not have any affect on IPv4 sockets.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// socket.set_multicast_loop_v6(false).expect("set_multicast_loop_v6 call failed");
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn set_multicast_loop_v6(&self, multicast_loop_v6: bool) -> io::Result<()> {
        self.0.set_multicast_loop_v6(multicast_loop_v6)
    }

    /// Gets the value of the `IPV6_MULTICAST_LOOP` option for this socket.
    ///
    /// For more information about this option, see
    /// [`set_multicast_loop_v6`][link].
    ///
    /// [link]: #method.set_multicast_loop_v6
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// socket.set_multicast_loop_v6(false).expect("set_multicast_loop_v6 call failed");
    /// assert_eq!(socket.multicast_loop_v6().unwrap(), false);
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn multicast_loop_v6(&self) -> io::Result<bool> {
        self.0.multicast_loop_v6()
    }

    /// Sets the value for the `IP_TTL` option on this socket.
    ///
    /// This value sets the time-to-live field that is used in every packet sent
    /// from this socket.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// socket.set_ttl(42).expect("set_ttl call failed");
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        self.0.set_ttl(ttl)
    }

    /// Gets the value of the `IP_TTL` option for this socket.
    ///
    /// For more information about this option, see [`set_ttl`][link].
    ///
    /// [link]: #method.set_ttl
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// socket.set_ttl(42).expect("set_ttl call failed");
    /// assert_eq!(socket.ttl().unwrap(), 42);
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn ttl(&self) -> io::Result<u32> {
        self.0.ttl()
    }

    /// Executes an operation of the `IP_ADD_MEMBERSHIP` type.
    ///
    /// This function specifies a new multicast group for this socket to join.
    /// The address must be a valid multicast address, and `interface` is the
    /// address of the local interface with which the system should join the
    /// multicast group. If it's equal to `INADDR_ANY` then an appropriate
    /// interface is chosen by the system.
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn join_multicast_v4(&self, multiaddr: &Ipv4Addr, interface: &Ipv4Addr) -> io::Result<()> {
        self.0.join_multicast_v4(multiaddr, interface)
    }

    /// Executes an operation of the `IPV6_ADD_MEMBERSHIP` type.
    ///
    /// This function specifies a new multicast group for this socket to join.
    /// The address must be a valid multicast address, and `interface` is the
    /// index of the interface to join/leave (or 0 to indicate any interface).
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn join_multicast_v6(&self, multiaddr: &Ipv6Addr, interface: u32) -> io::Result<()> {
        self.0.join_multicast_v6(multiaddr, interface)
    }

    /// Executes an operation of the `IP_DROP_MEMBERSHIP` type.
    ///
    /// For more information about this option, see
    /// [`join_multicast_v4`][link].
    ///
    /// [link]: #method.join_multicast_v4
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn leave_multicast_v4(&self, multiaddr: &Ipv4Addr, interface: &Ipv4Addr) -> io::Result<()> {
        self.0.leave_multicast_v4(multiaddr, interface)
    }

    /// Executes an operation of the `IPV6_DROP_MEMBERSHIP` type.
    ///
    /// For more information about this option, see
    /// [`join_multicast_v6`][link].
    ///
    /// [link]: #method.join_multicast_v6
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn leave_multicast_v6(&self, multiaddr: &Ipv6Addr, interface: u32) -> io::Result<()> {
        self.0.leave_multicast_v6(multiaddr, interface)
    }

    /// Get the value of the `SO_ERROR` option on this socket.
    ///
    /// This will retrieve the stored error in the underlying socket, clearing
    /// the field in the process. This can be useful for checking errors between
    /// calls.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// match socket.take_error() {
    ///     Ok(Some(error)) => println!("UdpSocket error: {:?}", error),
    ///     Ok(None) => println!("No error"),
    ///     Err(error) => println!("UdpSocket.take_error failed: {:?}", error),
    /// }
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        self.0.take_error()
    }

    /// Connects this UDP socket to a remote address, allowing the `send` and
    /// `recv` syscalls to be used to send data and also applies filters to only
    /// receive data from the specified address.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// socket.connect("127.0.0.1:8080").expect("connect function failed");
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn connect<A: ToSocketAddrs>(&self, addr: A) -> io::Result<()> {
        super::each_addr(addr, |addr| self.0.connect(addr))
    }

    /// Sends data on the socket to the remote address to which it is connected.
    ///
    /// The [`connect()`] method will connect this socket to a remote address. This
    /// method will fail if the socket is not connected.
    ///
    /// [`connect()`]: #method.connect
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// socket.connect("127.0.0.1:8080").expect("connect function failed");
    /// socket.send(&[0, 1, 2]).expect("couldn't send message");
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn send(&self, buf: &[u8]) -> io::Result<usize> {
        self.0.send(buf)
    }

    /// Receives data on the socket from the remote address to which it is
    /// connected.
    ///
    /// The `connect` method will connect this socket to a remote address. This
    /// method will fail if the socket is not connected.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// socket.connect("127.0.0.1:8080").expect("connect function failed");
    /// let mut buf = [0; 10];
    /// match socket.recv(&mut buf) {
    ///     Ok(received) => println!("received {} bytes", received),
    ///     Err(e) => println!("recv function failed: {:?}", e),
    /// }
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn recv(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.recv(buf)
    }

    /// Moves this UDP socket into or out of nonblocking mode.
    ///
    /// On Unix this corresponds to calling fcntl, and on Windows this
    /// corresponds to calling ioctlsocket.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::net::UdpSocket;
    ///
    /// let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    /// socket.set_nonblocking(true).expect("set_nonblocking call failed");
    /// ```
    #[stable(feature = "net2_mutators", since = "1.9.0")]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.0.set_nonblocking(nonblocking)
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

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for UdpSocket {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(all(test, not(target_os = "emscripten")))]
mod tests {
    use io::ErrorKind;
    use net::*;
    use net::test::{next_test_ip4, next_test_ip6};
    use sync::mpsc::channel;
    use sys_common::AsInner;
    use time::{Instant, Duration};
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

        let stream = t!(UdpSocket::bind(&addr));
        t!(stream.set_read_timeout(Some(Duration::from_millis(1000))));

        let mut buf = [0; 10];

        let start = Instant::now();
        let kind = stream.recv_from(&mut buf).err().expect("expected error").kind();
        assert!(kind == ErrorKind::WouldBlock || kind == ErrorKind::TimedOut);
        assert!(start.elapsed() > Duration::from_millis(400));
    }

    #[test]
    fn test_read_with_timeout() {
        let addr = next_test_ip4();

        let stream = t!(UdpSocket::bind(&addr));
        t!(stream.set_read_timeout(Some(Duration::from_millis(1000))));

        t!(stream.send_to(b"hello world", &addr));

        let mut buf = [0; 11];
        t!(stream.recv_from(&mut buf));
        assert_eq!(b"hello world", &buf[..]);

        let start = Instant::now();
        let kind = stream.recv_from(&mut buf).err().expect("expected error").kind();
        assert!(kind == ErrorKind::WouldBlock || kind == ErrorKind::TimedOut);
        assert!(start.elapsed() > Duration::from_millis(400));
    }

    #[test]
    fn connect_send_recv() {
        let addr = next_test_ip4();

        let socket = t!(UdpSocket::bind(&addr));
        t!(socket.connect(addr));

        t!(socket.send(b"hello world"));

        let mut buf = [0; 11];
        t!(socket.recv(&mut buf));
        assert_eq!(b"hello world", &buf[..]);
    }

    #[test]
    fn ttl() {
        let ttl = 100;

        let addr = next_test_ip4();

        let stream = t!(UdpSocket::bind(&addr));

        t!(stream.set_ttl(ttl));
        assert_eq!(ttl, t!(stream.ttl()));
    }

    #[test]
    fn set_nonblocking() {
        each_ip(&mut |addr, _| {
            let socket = t!(UdpSocket::bind(&addr));

            t!(socket.set_nonblocking(true));
            t!(socket.set_nonblocking(false));

            t!(socket.connect(addr));

            t!(socket.set_nonblocking(false));
            t!(socket.set_nonblocking(true));

            let mut buf = [0];
            match socket.recv(&mut buf) {
                Ok(_) => panic!("expected error"),
                Err(ref e) if e.kind() == ErrorKind::WouldBlock => {}
                Err(e) => panic!("unexpected error {}", e),
            }
        })
    }
}
