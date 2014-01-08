// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
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
use io::net::ip::{IpAddr, SocketAddr};
use io::{io_error, EndOfFile};
use rt::rtio::{IoFactory, LocalIo, RtioRawSocket, Protocol, CommDomain};

pub struct RawSocket {
    priv obj: ~RtioRawSocket
}

impl RawSocket {
    pub fn new(domain: CommDomain, protocol: Protocol, includeIpHeader: bool) -> Option<RawSocket> {
        LocalIo::maybe_raise(|io| {
            io.raw_socket_new(domain, protocol, includeIpHeader).map(|s| RawSocket { obj: s })
        })
    }

    pub fn recvfrom(&mut self, buf: &mut [u8]) -> Option<(uint, SocketAddr)> {
        match self.obj.recvfrom(buf) {
            Ok((nread, src)) => Some((nread, src)),
            Err(ioerr) => {
                // EOF is indicated by returning None
                if ioerr.kind != EndOfFile {
                    io_error::cond.raise(ioerr);
                }
                None
            }
        }
    }

    pub fn sendto(&mut self, buf: &[u8], dst: IpAddr) {
        match self.obj.sendto(buf, dst) {
            Ok(_) => (),
            Err(ioerr) => io_error::cond.raise(ioerr),
        }
    }
}
