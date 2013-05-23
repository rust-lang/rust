// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::*;
use super::super::*;
use super::ip::IpAddr;

pub struct UdpStream;

impl UdpStream {
    pub fn connect(_addr: IpAddr) -> Option<UdpStream> {
        fail!()
    }
}

impl Reader for UdpStream {
    fn read(&mut self, _buf: &mut [u8]) -> Option<uint> { fail!() }

    fn eof(&mut self) -> bool { fail!() }
}

impl Writer for UdpStream {
    fn write(&mut self, _buf: &[u8]) { fail!() }

    fn flush(&mut self) { fail!() }
}

pub struct UdpListener;

impl UdpListener {
    pub fn bind(_addr: IpAddr) -> Option<UdpListener> {
        fail!()
    }
}

impl Listener<UdpStream> for UdpListener {
    fn accept(&mut self) -> Option<UdpStream> { fail!() }
}
