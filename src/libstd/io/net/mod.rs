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
use ops::FnMut;
use option::Option::None;
use result::Result::{Ok, Err};
use self::ip::{SocketAddr, ToSocketAddr};

pub use self::addrinfo::get_host_addresses;

pub mod addrinfo;
pub mod tcp;
pub mod udp;
pub mod ip;
pub mod pipe;

fn with_addresses<A, T, F>(addr: A, mut action: F) -> IoResult<T> where
    A: ToSocketAddr,
    F: FnMut(SocketAddr) -> IoResult<T>,
{
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
