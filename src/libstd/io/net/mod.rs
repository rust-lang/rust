// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Networking I/O

use io::{IoError, IoResult, InvalidInput};
use option::None;
use result::{Result, Ok, Err};
use rt::rtio;
use self::ip::{Ipv4Addr, Ipv6Addr, IpAddr, SocketAddr, ToSocketAddr};

pub use self::addrinfo::get_host_addresses;

pub mod addrinfo;
pub mod tcp;
pub mod udp;
pub mod ip;
pub mod pipe;

fn to_rtio(ip: IpAddr) -> rtio::IpAddr {
    match ip {
        Ipv4Addr(a, b, c, d) => rtio::Ipv4Addr(a, b, c, d),
        Ipv6Addr(a, b, c, d, e, f, g, h) => {
            rtio::Ipv6Addr(a, b, c, d, e, f, g, h)
        }
    }
}

fn from_rtio(ip: rtio::IpAddr) -> IpAddr {
    match ip {
        rtio::Ipv4Addr(a, b, c, d) => Ipv4Addr(a, b, c, d),
        rtio::Ipv6Addr(a, b, c, d, e, f, g, h) => {
            Ipv6Addr(a, b, c, d, e, f, g, h)
        }
    }
}

fn with_addresses_io<A: ToSocketAddr, T>(
    addr: A,
    action: |&mut rtio::IoFactory, rtio::SocketAddr| -> Result<T, rtio::IoError>
) -> Result<T, IoError> {
    const DEFAULT_ERROR: IoError = IoError {
        kind: InvalidInput,
        desc: "no addresses found for hostname",
        detail: None
    };

    let addresses = try!(addr.to_socket_addr_all());
    let mut err = DEFAULT_ERROR;
    for addr in addresses.into_iter() {
        let addr = rtio::SocketAddr { ip: to_rtio(addr.ip), port: addr.port };
        match rtio::LocalIo::maybe_raise(|io| action(io, addr)) {
            Ok(r) => return Ok(r),
            Err(e) => err = IoError::from_rtio_error(e)
        }
    }
    Err(err)
}

fn with_addresses<A: ToSocketAddr, T>(addr: A, action: |SocketAddr| -> IoResult<T>)
    -> IoResult<T> {
    const DEFAULT_ERROR: IoError = IoError {
        kind: InvalidInput,
        desc: "no addresses found for hostname",
        detail: None
    };

    let addresses = try!(addr.to_socket_addr_all());
    let mut err = DEFAULT_ERROR;
    for addr in addresses.into_iter() {
        match action(addr) {
            Ok(r) => return Ok(r),
            Err(e) => err = e
        }
    }
    Err(err)
}
