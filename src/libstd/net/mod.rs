//! Networking primitives for TCP/UDP communication.
//!
//! This module provides networking functionality for the Transmission Control and User
//! Datagram Protocols, as well as types for IP and socket addresses.
//!
//! # Organization
//!
//! * [`TcpListener`] and [`TcpStream`] provide functionality for communication over TCP
//! * [`UdpSocket`] provides functionality for communication over UDP
//! * [`IpAddr`] represents IP addresses of either IPv4 or IPv6; [`Ipv4Addr`] and
//!   [`Ipv6Addr`] are respectively IPv4 and IPv6 addresses
//! * [`SocketAddr`] represents socket addresses of either IPv4 or IPv6; [`SocketAddrV4`]
//!   and [`SocketAddrV6`] are respectively IPv4 and IPv6 socket addresses
//! * [`ToSocketAddrs`] is a trait that used for generic address resolution when interacting
//!   with networking objects like [`TcpListener`], [`TcpStream`] or [`UdpSocket`]
//! * Other types are return or parameter types for various methods in this module
//!
//! [`IpAddr`]: ../../std/net/enum.IpAddr.html
//! [`Ipv4Addr`]: ../../std/net/struct.Ipv4Addr.html
//! [`Ipv6Addr`]: ../../std/net/struct.Ipv6Addr.html
//! [`SocketAddr`]: ../../std/net/enum.SocketAddr.html
//! [`SocketAddrV4`]: ../../std/net/struct.SocketAddrV4.html
//! [`SocketAddrV6`]: ../../std/net/struct.SocketAddrV6.html
//! [`TcpListener`]: ../../std/net/struct.TcpListener.html
//! [`TcpStream`]: ../../std/net/struct.TcpStream.html
//! [`ToSocketAddrs`]: ../../std/net/trait.ToSocketAddrs.html
//! [`UdpSocket`]: ../../std/net/struct.UdpSocket.html

#![stable(feature = "rust1", since = "1.0.0")]

use crate::io::{self, Error, ErrorKind};

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::addr::{SocketAddr, SocketAddrV4, SocketAddrV6, ToSocketAddrs};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::ip::{IpAddr, Ipv4Addr, Ipv6Addr, Ipv6MulticastScope};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::parser::AddrParseError;
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::tcp::{Incoming, TcpListener, TcpStream};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::udp::UdpSocket;

mod addr;
mod ip;
mod parser;
mod tcp;
#[cfg(test)]
mod test;
mod udp;

/// Possible values which can be passed to the [`shutdown`] method of
/// [`TcpStream`].
///
/// [`shutdown`]: struct.TcpStream.html#method.shutdown
/// [`TcpStream`]: struct.TcpStream.html
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum Shutdown {
    /// The reading portion of the [`TcpStream`] should be shut down.
    ///
    /// All currently blocked and future [reads] will return [`Ok(0)`].
    ///
    /// [`TcpStream`]: ../../std/net/struct.TcpStream.html
    /// [reads]: ../../std/io/trait.Read.html
    /// [`Ok(0)`]: ../../std/result/enum.Result.html#variant.Ok
    #[stable(feature = "rust1", since = "1.0.0")]
    Read,
    /// The writing portion of the [`TcpStream`] should be shut down.
    ///
    /// All currently blocked and future [writes] will return an error.
    ///
    /// [`TcpStream`]: ../../std/net/struct.TcpStream.html
    /// [writes]: ../../std/io/trait.Write.html
    #[stable(feature = "rust1", since = "1.0.0")]
    Write,
    /// Both the reading and the writing portions of the [`TcpStream`] should be shut down.
    ///
    /// See [`Shutdown::Read`] and [`Shutdown::Write`] for more information.
    ///
    /// [`TcpStream`]: ../../std/net/struct.TcpStream.html
    /// [`Shutdown::Read`]: #variant.Read
    /// [`Shutdown::Write`]: #variant.Write
    #[stable(feature = "rust1", since = "1.0.0")]
    Both,
}

#[inline]
const fn htons(i: u16) -> u16 {
    i.to_be()
}
#[inline]
const fn ntohs(i: u16) -> u16 {
    u16::from_be(i)
}

fn each_addr<A: ToSocketAddrs, F, T>(addr: A, mut f: F) -> io::Result<T>
where
    F: FnMut(io::Result<&SocketAddr>) -> io::Result<T>,
{
    let addrs = match addr.to_socket_addrs() {
        Ok(addrs) => addrs,
        Err(e) => return f(Err(e)),
    };
    let mut last_err = None;
    for addr in addrs {
        match f(Ok(&addr)) {
            Ok(l) => return Ok(l),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| {
        Error::new(ErrorKind::InvalidInput, "could not resolve to any addresses")
    }))
}
