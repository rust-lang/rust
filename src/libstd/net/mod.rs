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

#![stable(feature = "rust1", since = "1.0.0")]

use prelude::v1::*;
use sys::inner::*;
use sys::net as sys;
use sys::error;

use io::{self, Error, ErrorKind};

pub use self::ip::{IpAddr, Ipv4Addr, Ipv6Addr, Ipv6MulticastScope};
pub use self::addr::{SocketAddr, SocketAddrV4, SocketAddrV6, ToSocketAddrs};
pub use self::tcp::{TcpStream, TcpListener, Incoming};
pub use self::udp::UdpSocket;
pub use self::parser::AddrParseError;

mod ip;
mod addr;
mod tcp;
mod udp;
mod parser;
#[cfg(test)] mod test;

/// Possible values which can be passed to the `shutdown` method of `TcpStream`.
#[derive(Copy, Clone, PartialEq, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum Shutdown {
    /// Indicates that the reading portion of this stream/socket should be shut
    /// down. All currently blocked and future reads will return `Ok(0)`.
    #[stable(feature = "rust1", since = "1.0.0")]
    Read,
    /// Indicates that the writing portion of this stream/socket should be shut
    /// down. All currently blocked and future writes will return an error.
    #[stable(feature = "rust1", since = "1.0.0")]
    Write,
    /// Shut down both the reading and writing portions of this stream.
    ///
    /// See `Shutdown::Read` and `Shutdown::Write` for more information.
    #[stable(feature = "rust1", since = "1.0.0")]
    Both,
}

fn each_addr<A: ToSocketAddrs, F, T>(addr: A, mut f: F) -> io::Result<T>
    where F: FnMut(&sys::SocketAddr) -> error::Result<T>
{
    let mut last_err = None;
    for addr in try!(addr.to_socket_addrs()) {
        match f(&addr.into_inner()) {
            Ok(l) => return Ok(l),
            Err(e) => last_err = Some(e.into()),
        }
    }
    Err(last_err.unwrap_or_else(|| {
        Error::new(ErrorKind::InvalidInput,
                   "could not resolve to any addresses")
    }))
}

/// An iterator over `SocketAddr` values returned from a host lookup operation.
#[unstable(feature = "lookup_host", reason = "unsure about the returned \
                                              iterator and returning socket \
                                              addresses",
           issue = "27705")]
pub struct LookupHost(sys::LookupHost);

#[unstable(feature = "lookup_host", reason = "unsure about the returned \
                                              iterator and returning socket \
                                              addresses",
           issue = "27705")]
impl Iterator for LookupHost {
    type Item = io::Result<SocketAddr>;
    fn next(&mut self) -> Option<io::Result<SocketAddr>> { self.0.next().map(|i| i.map(FromInner::from_inner).map_err(From::from)) }
}

/// Resolve the host specified by `host` as a number of `SocketAddr` instances.
///
/// This method may perform a DNS query to resolve `host` and may also inspect
/// system configuration to resolve the specified hostname.
///
/// # Examples
///
/// ```no_run
/// #![feature(lookup_host)]
///
/// use std::net;
///
/// # fn foo() -> std::io::Result<()> {
/// for host in try!(net::lookup_host("rust-lang.org")) {
///     println!("found address: {}", try!(host));
/// }
/// # Ok(())
/// # }
/// ```
#[unstable(feature = "lookup_host", reason = "unsure about the returned \
                                              iterator and returning socket \
                                              addresses",
           issue = "27705")]
pub fn lookup_host(host: &str) -> io::Result<LookupHost> {
    sys::lookup_host(host).map(LookupHost).map_err(From::from)
}

/// Resolve the given address to a hostname.
///
/// This function may perform a DNS query to resolve `addr` and may also inspect
/// system configuration to resolve the specified address. If the address
/// cannot be resolved, it is returned in string format.
#[unstable(feature = "lookup_addr", reason = "recent addition",
           issue = "27705")]
pub fn lookup_addr(addr: &IpAddr) -> io::Result<String> {
    sys::lookup_addr(&addr.into_inner()).map(|a| String::from_utf8_lossy(a.as_bytes()).into_owned()).map_err(From::from)
}
