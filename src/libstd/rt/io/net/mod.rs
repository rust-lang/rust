// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use option::{Option, Some, None};
use result::{Ok, Err};
use rt::io::io_error;
use rt::io::net::ip::IpAddr;
use rt::rtio::{IoFactory, IoFactoryObject};
use rt::local::Local;

pub mod tcp;
pub mod udp;
pub mod ip;
#[cfg(unix)]
pub mod unix;

/// Simplistic name resolution
pub fn get_host_addresses(host: &str) -> Option<~[IpAddr]> {
    /*!
     * Get the IP addresses for a given host name.
     *
     * Raises io_error on failure.
     */

    let ipaddrs = unsafe {
        let io: *mut IoFactoryObject = Local::unsafe_borrow();
        (*io).get_host_addresses(host)
    };

    match ipaddrs {
        Ok(i) => Some(i),
        Err(ioerr) => {
            io_error::cond.raise(ioerr);
            None
        }
    }
}

#[cfg(test)]
mod test {
    use option::Some;
    use rt::io::net::ip::Ipv4Addr;
    use super::*;

    #[test]
    fn dns_smoke_test() {
        let ipaddrs = get_host_addresses("localhost").unwrap();
        let mut found_local = false;
        let local_addr = &Ipv4Addr(127, 0, 0, 1);
        for addr in ipaddrs.iter() {
            found_local = found_local || addr == local_addr;
        }
        assert!(found_local);
    }
}
