// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Internet Protocol (IP) addresses.
//!
//! This module contains functions useful for parsing, formatting, and
//! manipulating IP addresses.

#![allow(missing_docs)]

pub use self::IpAddr::*;

use fmt;
use io::{mod, IoResult, IoError};
use io::net;
use iter::Iterator;
use option::{Option, None, Some};
use result::{Ok, Err};
use str::{FromStr, StrPrelude};
use slice::{CloneSlicePrelude, SlicePrelude};
use vec::Vec;

pub type Port = u16;

#[deriving(PartialEq, Eq, Clone, Hash)]
pub enum IpAddr {
    Ipv4Addr(u8, u8, u8, u8),
    Ipv6Addr(u16, u16, u16, u16, u16, u16, u16, u16)
}

impl fmt::Show for IpAddr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result { unimplemented!() }
}

#[deriving(PartialEq, Eq, Clone, Hash)]
pub struct SocketAddr {
    pub ip: IpAddr,
    pub port: Port,
}

impl fmt::Show for SocketAddr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { unimplemented!() }
}

struct Parser<'a> {
    // parsing as ASCII, so can use byte array
    s: &'a [u8],
    pos: uint,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Parser<'a> { unimplemented!() }

    fn is_eof(&self) -> bool { unimplemented!() }

    // Commit only if parser returns Some
    fn read_atomically<T>(&mut self, cb: |&mut Parser| -> Option<T>)
                       -> Option<T> { unimplemented!() }

    // Commit only if parser read till EOF
    fn read_till_eof<T>(&mut self, cb: |&mut Parser| -> Option<T>)
                     -> Option<T> { unimplemented!() }

    // Return result of first successful parser
    fn read_or<T>(&mut self, parsers: &mut [|&mut Parser| -> Option<T>])
               -> Option<T> { unimplemented!() }

    // Apply 3 parsers sequentially
    fn read_seq_3<A,
                  B,
                  C>(
                  &mut self,
                  pa: |&mut Parser| -> Option<A>,
                  pb: |&mut Parser| -> Option<B>,
                  pc: |&mut Parser| -> Option<C>)
                  -> Option<(A, B, C)> { unimplemented!() }

    // Read next char
    fn read_char(&mut self) -> Option<char> { unimplemented!() }

    // Return char and advance iff next char is equal to requested
    fn read_given_char(&mut self, c: char) -> Option<char> { unimplemented!() }

    // Read digit
    fn read_digit(&mut self, radix: u8) -> Option<u8> { unimplemented!() }

    fn read_number_impl(&mut self, radix: u8, max_digits: u32, upto: u32) -> Option<u32> { unimplemented!() }

    // Read number, failing if max_digits of number value exceeded
    fn read_number(&mut self, radix: u8, max_digits: u32, upto: u32) -> Option<u32> { unimplemented!() }

    fn read_ipv4_addr_impl(&mut self) -> Option<IpAddr> { unimplemented!() }

    // Read IPv4 address
    fn read_ipv4_addr(&mut self) -> Option<IpAddr> { unimplemented!() }

    fn read_ipv6_addr_impl(&mut self) -> Option<IpAddr> { unimplemented!() }

    fn read_ipv6_addr(&mut self) -> Option<IpAddr> { unimplemented!() }

    fn read_ip_addr(&mut self) -> Option<IpAddr> { unimplemented!() }

    fn read_socket_addr(&mut self) -> Option<SocketAddr> { unimplemented!() }
}

impl FromStr for IpAddr {
    fn from_str(s: &str) -> Option<IpAddr> { unimplemented!() }
}

impl FromStr for SocketAddr {
    fn from_str(s: &str) -> Option<SocketAddr> { unimplemented!() }
}

/// A trait for objects which can be converted or resolved to one or more `SocketAddr` values.
///
/// Implementing types minimally have to implement either `to_socket_addr` or `to_socket_addr_all`
/// method, and its trivial counterpart will be available automatically.
///
/// This trait is used for generic address resolution when constructing network objects.
/// By default it is implemented for the following types:
///
///  * `SocketAddr` - `to_socket_addr` is identity function.
///
///  * `(IpAddr, u16)` - `to_socket_addr` constructs `SocketAddr` trivially.
///
///  * `(&str, u16)` - the string should be either a string representation of an IP address
///    expected by `FromStr` implementation for `IpAddr` or a host name.
///
///    For the former, `to_socket_addr_all` returns a vector with a single element corresponding
///    to that IP address joined with the given port.
///
///    For the latter, it tries to resolve the host name and returns a vector of all IP addresses
///    for the host name, each joined with the given port.
///
///  * `&str` - the string should be either a string representation of a `SocketAddr` as
///    expected by its `FromStr` implementation or a string like `<host_name>:<port>` pair
///    where `<port>` is a `u16` value.
///
///    For the former, `to_socker_addr_all` returns a vector with a single element corresponding
///    to that socker address.
///
///    For the latter, it tries to resolve the host name and returns a vector of all IP addresses
///    for the host name, each joined with the port.
///
///
/// This trait allows constructing network objects like `TcpStream` or `UdpSocket` easily with
/// values of various types for the bind/connection address. It is needed because sometimes
/// one type is more appropriate than the other: for simple uses a string like `"localhost:12345"`
/// is much nicer than manual construction of the corresponding `SocketAddr`, but sometimes
/// `SocketAddr` value is *the* main source of the address, and converting it to some other type
/// (e.g. a string) just for it to be converted back to `SocketAddr` in constructor methods
/// is pointless.
///
/// Some examples:
///
/// ```rust,no_run
/// # #![allow(unused_must_use)]
///
/// use std::io::{TcpStream, TcpListener};
/// use std::io::net::udp::UdpSocket;
/// use std::io::net::ip::{Ipv4Addr, SocketAddr};
///
/// fn main() {
///     // The following lines are equivalent modulo possible "localhost" name resolution
///     // differences
///     let tcp_s = TcpStream::connect(SocketAddr { ip: Ipv4Addr(127, 0, 0, 1), port: 12345 });
///     let tcp_s = TcpStream::connect((Ipv4Addr(127, 0, 0, 1), 12345u16));
///     let tcp_s = TcpStream::connect(("127.0.0.1", 12345u16));
///     let tcp_s = TcpStream::connect(("localhost", 12345u16));
///     let tcp_s = TcpStream::connect("127.0.0.1:12345");
///     let tcp_s = TcpStream::connect("localhost:12345");
///
///     // TcpListener::bind(), UdpSocket::bind() and UdpSocket::send_to() behave similarly
///     let tcp_l = TcpListener::bind("localhost:12345");
///
///     let mut udp_s = UdpSocket::bind(("127.0.0.1", 23451u16)).unwrap();
///     udp_s.send_to([7u8, 7u8, 7u8].as_slice(), (Ipv4Addr(127, 0, 0, 1), 23451u16));
/// }
/// ```
pub trait ToSocketAddr {
    /// Converts this object to single socket address value.
    ///
    /// If more than one value is available, this method returns the first one. If no
    /// values are available, this method returns an `IoError`.
    ///
    /// By default this method delegates to `to_socket_addr_all` method, taking the first
    /// item from its result.
    fn to_socket_addr(&self) -> IoResult<SocketAddr> { unimplemented!() }

    /// Converts this object to all available socket address values.
    ///
    /// Some values like host name string naturally corrrespond to multiple IP addresses.
    /// This method tries to return all available addresses corresponding to this object.
    ///
    /// By default this method delegates to `to_socket_addr` method, creating a singleton
    /// vector from its result.
    #[inline]
    fn to_socket_addr_all(&self) -> IoResult<Vec<SocketAddr>> { unimplemented!() }
}

impl ToSocketAddr for SocketAddr {
    #[inline]
    fn to_socket_addr(&self) -> IoResult<SocketAddr> { unimplemented!() }
}

impl ToSocketAddr for (IpAddr, u16) {
    #[inline]
    fn to_socket_addr(&self) -> IoResult<SocketAddr> { unimplemented!() }
}

fn resolve_socket_addr(s: &str, p: u16) -> IoResult<Vec<SocketAddr>> { unimplemented!() }

fn parse_and_resolve_socket_addr(s: &str) -> IoResult<Vec<SocketAddr>> { unimplemented!() }

impl<'a> ToSocketAddr for (&'a str, u16) {
    fn to_socket_addr_all(&self) -> IoResult<Vec<SocketAddr>> { unimplemented!() }
}

// accepts strings like 'localhost:12345'
impl<'a> ToSocketAddr for &'a str {
    fn to_socket_addr(&self) -> IoResult<SocketAddr> { unimplemented!() }

    fn to_socket_addr_all(&self) -> IoResult<Vec<SocketAddr>> { unimplemented!() }
}
