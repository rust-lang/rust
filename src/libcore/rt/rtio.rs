// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use option::*;
use result::*;

use super::io::net::ip::IpAddr;

// XXX: ~object doesn't work currently so these are some placeholder
// types to use instead
pub type EventLoopObject = super::uvio::UvEventLoop;
pub type IoFactoryObject = super::uvio::UvIoFactory;
pub type StreamObject = super::uvio::UvStream;
pub type TcpListenerObject = super::uvio::UvTcpListener;

pub trait EventLoop {
    fn run(&mut self);
    fn callback(&mut self, ~fn());
    /// The asynchronous I/O services. Not all event loops may provide one
    fn io<'a>(&'a mut self) -> Option<&'a mut IoFactoryObject>;
}

pub trait IoFactory {
    fn connect(&mut self, addr: IpAddr) -> Option<~StreamObject>;
    fn bind(&mut self, addr: IpAddr) -> Option<~TcpListenerObject>;
}

pub trait TcpListener {
    fn listen(&mut self) -> Option<~StreamObject>;
}

pub trait Stream {
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, ()>;
    fn write(&mut self, buf: &[u8]) -> Result<(), ()>;
}
