// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io::net::ip::{IpAddr};
use io::{IoResult};
use rt::rtio::{IoFactory, LocalIo, RtioRawSocket};

pub struct RawSocket {
    priv obj: ~RtioRawSocket
}

impl RawSocket {
    pub fn new(domain: i32, protocol: i32, includeIpHeader: bool) -> IoResult<RawSocket> {
        LocalIo::maybe_raise(|io| {
            io.raw_socket_new(domain, protocol, includeIpHeader).map(|s| RawSocket { obj: s })
        })
    }

    pub fn recvfrom(&mut self, buf: &mut [u8]) -> IoResult<(uint, IpAddr)> {
        self.obj.recvfrom(buf)
    }

    pub fn sendto(&mut self, buf: &[u8], dst: IpAddr) -> IoResult<int> {
        self.obj.sendto(buf, dst)
    }
}
