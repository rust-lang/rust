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
use io::net::ip::{SocketAddr, IpAddr, ToSocketAddr};
use io::{Reader, Writer, IoResult};
use option::Option;
use result::{Ok, Err};
use sys::udp::UdpSocket as UdpSocketImp;

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
/// #![feature(slicing_syntax)]
///
/// use std::io::net::udp::UdpSocket;
/// use std::io::net::ip::{Ipv4Addr, SocketAddr};
/// fn main() {
///     let addr = SocketAddr { ip: Ipv4Addr(127, 0, 0, 1), port: 34254 };
///     let mut socket = match UdpSocket::bind(addr) {
///         Ok(s) => s,
///         Err(e) => panic!("couldn't bind socket: {}", e),
///     };
///
///     let mut buf = [0, ..10];
///     match socket.recv_from(&mut buf) {
///         Ok((amt, src)) => {
///             // Send a reply to the socket we received data from
///             let buf = buf[mut ..amt];
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
    pub fn bind<A: ToSocketAddr>(addr: A) -> IoResult<UdpSocket> { unimplemented!() }

    /// Receives data from the socket. On success, returns the number of bytes
    /// read and the address from whence the data came.
    pub fn recv_from(&mut self, buf: &mut [u8]) -> IoResult<(uint, SocketAddr)> { unimplemented!() }

    /// Sends data on the socket to the given address. Returns nothing on
    /// success.
    ///
    /// Address type can be any implementor of `ToSocketAddr` trait. See its
    /// documentation for concrete examples.
    pub fn send_to<A: ToSocketAddr>(&mut self, buf: &[u8], addr: A) -> IoResult<()> { unimplemented!() }

    /// Creates a `UdpStream`, which allows use of the `Reader` and `Writer`
    /// traits to receive and send data from the same address. This transfers
    /// ownership of the socket to the stream.
    ///
    /// Note that this call does not perform any actual network communication,
    /// because UDP is a datagram protocol.
    #[deprecated = "`UdpStream` has been deprecated"]
    #[allow(deprecated)]
    pub fn connect(self, other: SocketAddr) -> UdpStream { unimplemented!() }

    /// Returns the socket address that this socket was created from.
    pub fn socket_name(&mut self) -> IoResult<SocketAddr> { unimplemented!() }

    /// Joins a multicast IP address (becomes a member of it)
    #[experimental]
    pub fn join_multicast(&mut self, multi: IpAddr) -> IoResult<()> { unimplemented!() }

    /// Leaves a multicast IP address (drops membership from it)
    #[experimental]
    pub fn leave_multicast(&mut self, multi: IpAddr) -> IoResult<()> { unimplemented!() }

    /// Set the multicast loop flag to the specified value
    ///
    /// This lets multicast packets loop back to local sockets (if enabled)
    #[experimental]
    pub fn set_multicast_loop(&mut self, on: bool) -> IoResult<()> { unimplemented!() }

    /// Sets the multicast TTL
    #[experimental]
    pub fn set_multicast_ttl(&mut self, ttl: int) -> IoResult<()> { unimplemented!() }

    /// Sets this socket's TTL
    #[experimental]
    pub fn set_ttl(&mut self, ttl: int) -> IoResult<()> { unimplemented!() }

    /// Sets the broadcast flag on or off
    #[experimental]
    pub fn set_broadcast(&mut self, broadcast: bool) -> IoResult<()> { unimplemented!() }

    /// Sets the read/write timeout for this socket.
    ///
    /// For more information, see `TcpStream::set_timeout`
    #[experimental = "the timeout argument may change in type and value"]
    pub fn set_timeout(&mut self, timeout_ms: Option<u64>) { unimplemented!() }

    /// Sets the read timeout for this socket.
    ///
    /// For more information, see `TcpStream::set_timeout`
    #[experimental = "the timeout argument may change in type and value"]
    pub fn set_read_timeout(&mut self, timeout_ms: Option<u64>) { unimplemented!() }

    /// Sets the write timeout for this socket.
    ///
    /// For more information, see `TcpStream::set_timeout`
    #[experimental = "the timeout argument may change in type and value"]
    pub fn set_write_timeout(&mut self, timeout_ms: Option<u64>) { unimplemented!() }
}

impl Clone for UdpSocket {
    /// Creates a new handle to this UDP socket, allowing for simultaneous
    /// reads and writes of the socket.
    ///
    /// The underlying UDP socket will not be closed until all handles to the
    /// socket have been deallocated. Two concurrent reads will not receive
    /// the same data. Instead, the first read will receive the first packet
    /// received, and the second read will receive the second packet.
    fn clone(&self) -> UdpSocket { unimplemented!() }
}

/// A type that allows convenient usage of a UDP stream connected to one
/// address via the `Reader` and `Writer` traits.
///
/// # Note
///
/// This structure has been deprecated because `Reader` is a stream-oriented API but UDP
/// is a packet-oriented protocol. Every `Reader` method will read a whole packet and
/// throw all superfluous bytes away so that they are no longer available for further
/// method calls.
#[deprecated]
pub struct UdpStream {
    socket: UdpSocket,
    connected_to: SocketAddr
}

impl UdpStream {
    /// Allows access to the underlying UDP socket owned by this stream. This
    /// is useful to, for example, use the socket to send data to hosts other
    /// than the one that this stream is connected to.
    pub fn as_socket<T>(&mut self, f: |&mut UdpSocket| -> T) -> T { unimplemented!() }

    /// Consumes this UDP stream and returns out the underlying socket.
    pub fn disconnect(self) -> UdpSocket { unimplemented!() }
}

impl Reader for UdpStream {
    /// Returns the next non-empty message from the specified address.
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { unimplemented!() }
}

impl Writer for UdpStream {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { unimplemented!() }
}
