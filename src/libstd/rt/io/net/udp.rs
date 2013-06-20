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
use rt::io::net::ip::IpAddr;
use rt::io::{Reader, Writer};
use rt::io::{io_error, read_error, EndOfFile};
use rt::rtio::{RtioUdpSocketObject, RtioUdpSocket, IoFactory, IoFactoryObject};
use rt::local::Local;

pub struct UdpSocket {
    rtsocket: ~RtioUdpSocketObject
}

impl UdpSocket {
    fn new(s: ~RtioUdpSocketObject) -> UdpSocket {
        UdpSocket { rtsocket: s }
    }

    pub fn bind(addr: IpAddr) -> Option<UdpSocket> {
        let socket = unsafe {
            let io = Local::unsafe_borrow::<IoFactoryObject>();
            (*io).udp_bind(addr)
        };
        match socket {
            Ok(s) => { Some(UdpSocket { rtsocket: s }) }
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
                return None;
            }
        }
    }

    pub fn recvfrom(&self, buf: &mut [u8]) -> Option<(uint, IpAddr)> {
        match (*self.rtsocket).recvfrom(buf) {
            Ok((nread, src)) => Some((nread, src)),
            Err(ioerr) => {
                // EOF is indicated by returning None
                // XXX do we ever find EOF reading UDP packets?
                if ioerr.kind != EndOfFile {
                    read_error::cond.raise(ioerr);
                }
                None
            }
        }
    }

    pub fn sendto(&self, buf: &[u8], dst: IpAddr) {
        match (*self.rtsocket).sendto(buf, dst) {
            Ok(_) => (),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
            }
        }
    }

    // XXX convert ~self to self eventually
    pub fn connect(~self, other: IpAddr) -> UdpStream {
        UdpStream { socket: self, connectedTo: other }
    }
}

pub struct UdpStream {
    socket: ~UdpSocket,
    connectedTo: IpAddr
}

impl UdpStream {
    pub fn as_socket<T>(&self, f: &fn(&UdpSocket) -> T) -> T {
        f(self.socket)
    }

    pub fn disconnect(self) -> ~UdpSocket {
        let UdpStream { socket: s, _ } = self;
        s
    }
}

impl Reader for UdpStream {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> { 
        let conn = self.connectedTo;
        do self.as_socket |sock| {
            sock.recvfrom(buf)
                .map_consume(|(nread,src)| if src == conn {nread} else {0})
        }
    }

    fn eof(&mut self) -> bool { fail!() }
}

impl Writer for UdpStream {
    fn write(&mut self, buf: &[u8]) {
        do self.as_socket |sock| {
            sock.sendto(buf, self.connectedTo);
        }
    }

    fn flush(&mut self) { fail!() }
}
