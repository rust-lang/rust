// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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

use io::{self, Error, ErrorKind};

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::ip::{IpAddr, Ipv4Addr, Ipv6Addr, Ipv6MulticastScope};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::addr::{SocketAddr, SocketAddrV4, SocketAddrV6, ToSocketAddrs};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::tcp::{TcpStream, TcpListener, Incoming};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::udp::UdpSocket;
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::parser::AddrParseError;

mod ip;
mod addr;
mod tcp;
mod udp;
mod parser;
#[cfg(test)]
mod test;

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

#[doc(hidden)]
trait NetInt {
    fn from_be(i: Self) -> Self;
    fn to_be(&self) -> Self;
}
macro_rules! doit {
    ($($t:ident)*) => ($(impl NetInt for $t {
        fn from_be(i: Self) -> Self { <$t>::from_be(i) }
        fn to_be(&self) -> Self { <$t>::to_be(*self) }
    })*)
}
doit! { i8 i16 i32 i64 isize u8 u16 u32 u64 usize }

fn hton<I: NetInt>(i: I) -> I { i.to_be() }
fn ntoh<I: NetInt>(i: I) -> I { I::from_be(i) }

fn each_addr<A: ToSocketAddrs, F, T>(addr: A, mut f: F) -> io::Result<T>
    where F: FnMut(&SocketAddr) -> io::Result<T>
{
    let mut last_err = None;
    for addr in addr.to_socket_addrs()? {
        match f(&addr) {
            Ok(l) => return Ok(l),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| {
        Error::new(ErrorKind::InvalidInput,
                   "could not resolve to any addresses")
    }))
}
