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
use super::*;
use super::super::*;
use super::ip::IpAddr;

pub struct TcpStream;

impl TcpStream {
    pub fn connect(_addr: IpAddr) -> Result<TcpStream, IoError> {
        fail!()
    }
}

impl Reader for TcpStream {
    fn read(&mut self, _buf: &mut [u8]) -> Option<uint> { fail!() }

    fn eof(&mut self) -> bool { fail!() }
}

impl Writer for TcpStream {
    fn write(&mut self, _buf: &[u8]) { fail!() }

    fn flush(&mut self) { fail!() }
}

impl Close for TcpStream {
    fn close(&mut self) { fail!() }
}

pub struct TcpListener;

impl TcpListener {
    pub fn new(_addr: IpAddr) -> TcpListener {
        fail!()
    }
}

impl Listener<TcpStream> for TcpListener {
    fn accept(&mut self) -> Option<TcpStream> { fail!() }
}
