// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Networking primitives for TCP/UDP communication
//!
//! > **NOTE**: This module is very much a work in progress and is under active
//! > development. At this time it is still recommended to use the `old_io`
//! > module while the details of this module shake out.

#![unstable(feature = "net")]

use prelude::v1::*;

use io::{self, Error, ErrorKind};
use num::Int;
use sys_common::net2 as net_imp;

pub use self::ip::{IpAddr, IpAddrFamily, Ipv4Addr, Ipv6Addr, Ipv6MulticastScope};
pub use self::addr::{SocketAddr, ToSocketAddrs};
pub use self::tcp::{TcpStream, TcpListener};
pub use self::udp::UdpSocket;

mod ip;
mod addr;
mod tcp;
mod udp;
mod parser;
#[cfg(test)] mod test;

/// Possible values which can be passed to the `shutdown` method of `TcpStream`
/// and `UdpSocket`.
#[derive(Copy, Clone, PartialEq)]
pub enum Shutdown {
    /// Indicates that the reading portion of this stream/socket should be shut
    /// down. All currently blocked and future reads will return `Ok(0)`.
    Read,
    /// Indicates that the writing portion of this stream/socket should be shut
    /// down. All currently blocked and future writes will return an error.
    Write,
    /// Shut down both the reading and writing portions of this stream.
    ///
    /// See `Shutdown::Read` and `Shutdown::Write` for more information.
    Both
}

fn hton<I: Int>(i: I) -> I { i.to_be() }
fn ntoh<I: Int>(i: I) -> I { Int::from_be(i) }

fn each_addr<A: ToSocketAddrs + ?Sized, F, T>(addr: &A, mut f: F) -> io::Result<T>
    where F: FnMut(&SocketAddr) -> io::Result<T>
{
    let mut last_err = None;
    for addr in try!(addr.to_socket_addrs()) {
        match f(&addr) {
            Ok(l) => return Ok(l),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| {
        Error::new(ErrorKind::InvalidInput,
                   "could not resolve to any addresses", None)
    }))
}

/// An iterator over `SocketAddr` values returned from a host lookup operation.
pub struct LookupHost(net_imp::LookupHost);

impl Iterator for LookupHost {
    type Item = io::Result<SocketAddr>;
    fn next(&mut self) -> Option<io::Result<SocketAddr>> { self.0.next() }
}

/// Resolve the host specified by `host` as a number of `SocketAddr` instances.
///
/// This method may perform a DNS query to resolve `host` and may also inspect
/// system configuration to resolve the specified hostname.
///
/// # Example
///
/// ```no_run
/// use std::net;
///
/// # fn foo() -> std::io::Result<()> {
/// for host in try!(net::lookup_host("rust-lang.org")) {
///     println!("found address: {}", try!(host));
/// }
/// # Ok(())
/// # }
/// ```
pub fn lookup_host(host: &str) -> io::Result<LookupHost> {
    net_imp::lookup_host(host).map(LookupHost)
}

#[cfg(test)]
mod test_of_this {
    use prelude::v1::*;
    use net;

    #[test]
    fn test_lookup_host() {
        let mut addrs = net::lookup_host("localhost").unwrap();
        assert!(addrs.any(|a| a.unwrap().ip() == net::IpAddr::new_v4(127, 0, 0, 1)));
    }

}
